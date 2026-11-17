"""Cache FP64 R4 log mass before page normalization; no precision reduction."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
BLOCK=256
WARPS=4
FUSED=False
@tr.jit
def local_mass(Z,O,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    a=tl.load(Z+4*i,i<N,0).to(tl.float64);b=tl.load(Z+4*i+1,i<N,0).to(tl.float64)
    c=tl.load(Z+4*i+2,i<N,0).to(tl.float64);d=tl.load(Z+4*i+3,i<N,0).to(tl.float64)
    m=tl.maximum(tl.maximum(tl.maximum(a,b),c),d)
    m=tl.where(tl.abs(m)==float('inf'),0.,m)
    v=lib.log(((lib.exp(a-m)+lib.exp(b-m))+lib.exp(c-m))+lib.exp(d-m))+m
    tl.store(O+i,v,i<N)
@tr.jit
def fused_mass(Z,O,P:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);p=tl.arange(0,B);i=row*P+p
    a=tl.load(Z+4*i,p<P,0).to(tl.float64);b=tl.load(Z+4*i+1,p<P,0).to(tl.float64)
    c=tl.load(Z+4*i+2,p<P,0).to(tl.float64);d=tl.load(Z+4*i+3,p<P,0).to(tl.float64)
    m=tl.maximum(tl.maximum(tl.maximum(a,b),c),d)
    m=tl.where(tl.abs(m)==float('inf'),0.,m)
    v=lib.log(((lib.exp(a-m)+lib.exp(b-m))+lib.exp(c-m))+lib.exp(d-m))+m
    v=tl.where(p<P,v,-float('inf'));mx=tl.max(v,0)
    total=tl.sum(lib.exp(v-mx),0);den=lib.log(total)+mx
    tl.store(O+i,v-den,p<P)
@torch.library.custom_op('streaming_sparse::cached_r4_mass',mutates_args=())
def local(z:torch.Tensor)->torch.Tensor:
    o=torch.empty(z.shape[:-1],dtype=torch.float64,device=z.device)
    local_mass[(tr.cdiv(o.numel(),BLOCK),)](z,o,o.numel(),BLOCK,num_warps=WARPS,enable_fp_fusion=False)
    return o
@local.register_fake
def _(z):return torch.empty(z.shape[:-1],dtype=torch.float64,device=z.device)
@torch.library.custom_op('streaming_sparse::fused_r4_mass',mutates_args=())
def fused(z:torch.Tensor)->torch.Tensor:
    o=torch.empty(z.shape[:-1],dtype=torch.float64,device=z.device);p=z.shape[-2]
    fused_mass[(o.numel()//p,)](z,o,p,tr.next_power_of_2(p),num_warps=WARPS,enable_fp_fusion=False)
    return o
@fused.register_fake
def _(z):return torch.empty(z.shape[:-1],dtype=torch.float64,device=z.device)
def normalized(z):
    if FUSED:return fused(z)
    lm=local(z)
    return lm-lm.logsumexp(-1,keepdim=True)
