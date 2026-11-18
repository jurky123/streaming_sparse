"""Read the same sorted token indices once for both full-width KV rows."""
import torch
import triton
import triton.language as tl

@triton.jit
def _pair(K,V,ORDER,KO,VO,N:tl.constexpr,KH:tl.constexpr,KT:tl.constexpr,
          VH:tl.constexpr,VT:tl.constexpr,B:tl.constexpr):
    h=tl.program_id(1)
    row=tl.program_id(0)*B+tl.arange(0,B)
    d=tl.arange(0,128)
    src=tl.load(ORDER+h*N+row,row<N,0).to(tl.int32)
    k=tl.load(K+h*KH+src[:,None]*KT+d[None,:],row[:,None]<N,0)
    v=tl.load(V+h*VH+src[:,None]*VT+d[None,:],row[:,None]<N,0)
    dest=(h*N+row[:,None])*128+d[None,:]
    tl.store(KO+dest,k,row[:,None]<N)
    tl.store(VO+dest,v,row[:,None]<N)

def gather_pair(k,v,order):
    h,n,d=k.shape
    assert d==128 and k.stride(-1)==v.stride(-1)==1 and order.is_contiguous()
    ko=torch.empty((h,n,d),device=k.device,dtype=k.dtype)
    vo=torch.empty_like(ko)
    _pair[(triton.cdiv(n,16),h)](k,v,order,ko,vo,n,*k.stride()[:2],*v.stride()[:2],16)
    return ko,vo
