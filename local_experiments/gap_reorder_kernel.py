"""Bit-preserving feature/K/V gather with one index load per token row."""
import torch
import triton as tr
import triton.language as tl
@tr.jit
def _reorder(X,ORDER,OUT,N:tl.constexpr,D:tl.constexpr,XH:tl.constexpr,XN:tl.constexpr,XD:tl.constexpr,OH:tl.constexpr,ON:tl.constexpr,B:tl.constexpr):
    h=tl.program_id(1);row=tl.program_id(0)*B+tl.arange(0,B);d=tl.arange(0,D)
    src=tl.load(ORDER+h*OH+row*ON,row<N,0)
    value=tl.load(X+h*XH+src[:,None]*XN+d[None,:]*XD,row[:,None]<N,0)
    tl.store(OUT+(h*N+row[:,None])*D+d[None,:],value,row[:,None]<N)
def reorder(x,order,block=16):
    h,n,d=x.shape
    assert d in (32,128) and order.shape==(h,n)
    out=torch.empty_like(x,memory_format=torch.contiguous_format)
    _reorder[(tr.cdiv(n,block),h)](x,order,out,n,d,*x.stride(),*order.stride(),block,num_warps=4)
    return out
