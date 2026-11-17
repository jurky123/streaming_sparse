"""Combine positive remove/add endpoint groups before their Cartesian product."""
import torch
def denominator(lb,rx,lr,ax,ea):
    ur=torch.logaddexp(torch.logaddexp(lb[:,:,None],rx),lr)
    va=cached_add(ax,ea) if CACHED_GROUPS else torch.logaddexp(ax,ea)
    return torch.logaddexp(ur[:,:,:,None],va[:,:,None,:])

# An explicit op boundary prevents Inductor from expanding V_a over all R actions.
import triton as tr
import triton.language as tl
from .scores_den import logadd
CACHED_GROUPS=False
@tr.jit
def add_kernel(X,Y,O,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,0).to(tl.float64);y=tl.load(Y+i,i<N,0).to(tl.float64)
    tl.store(O+i,logadd(x,y),i<N)
@torch.library.custom_op('streaming_sparse::endpoint_logadd',mutates_args=())
def cached_add(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    x=x.contiguous();y=y.contiguous();out=torch.empty_like(x)
    add_kernel[(tr.cdiv(x.numel(),256),)](x,y,out,x.numel(),256,num_warps=4,enable_fp_fusion=False)
    return out
@cached_add.register_fake
def _(x,y):return torch.empty_like(x)
