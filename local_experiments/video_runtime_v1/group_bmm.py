"""FP64 small grouped products for exclusion aggregates; no precision reduction."""
import torch
import triton as tr
import triton.language as tl
GROUP=1
WARPS=4
KERNEL_STATS={}
@tr.jit
def kernel(W,V,O,N:tl.constexpr,K:tl.constexpr,D:tl.constexpr,G:tl.constexpr):
    group=tl.program_id(0)*G+tl.arange(0,G);row=tl.program_id(1)
    k=tl.arange(0,K);d=tl.arange(0,D)
    w=tl.load(W+(group[:,None]*K+row)*K+k[None,:],group[:,None]<N,0)
    v=tl.load(V+(group[:,None,None]*K+k[None,:,None])*D+d[None,None,:],group[:,None,None]<N,0)
    out=tl.sum(w[:,:,None]*v,1)
    tl.store(O+(group[:,None]*K+row)*D+d[None,:],out,group[:,None]<N)
@torch.library.custom_op('streaming_sparse::group_bmm',mutates_args=())
def group_bmm(w:torch.Tensor,v:torch.Tensor)->torch.Tensor:
    k=w.shape[-1];d=v.shape[-1];n=w.numel()//(k*k);w=w.contiguous();v=v.contiguous()
    out=torch.empty((*w.shape[:-1],d),device=w.device,dtype=w.dtype)
    compiled=kernel[(tr.cdiv(n,GROUP),k)](w,v,out,n,k,d,GROUP,num_warps=WARPS,enable_fp_fusion=False)
    KERNEL_STATS[k]=dict(registers=compiled.n_regs,spills=compiled.n_spills,group=GROUP,warps=WARPS)
    return out
@group_bmm.register_fake
def _(w,v):return w.new_empty((*w.shape[:-1],v.shape[-1]))
