"""Centered variance and variable page-local log-count scoring."""
import torch,triton
import triton.language as tl
from .rollout_kernels import _pool
@triton.jit
def _score(M,V,K,A,C,LC,S,P:tl.constexpr,T:tl.constexpr,D:tl.constexpr,CENTER:tl.constexpr,COUNTS:tl.constexpr):
    m=tl.program_id(0)*16+tl.arange(0,16);n=tl.program_id(1)*64+tl.arange(0,64)
    h=tl.program_id(2);d=tl.arange(0,D)
    q=tl.load(M+(h*T+m[:,None])*D+d[None,:],m[:,None]<T,other=0)
    k=tl.load(K+(h*P*4+n[None,:])*D+d[:,None],n[None,:]<P*4,other=0)
    z=tl.dot(q,k,input_precision='tf32x3')*0.08838834764831845
    if COUNTS:
        count=tl.load(LC+h*P*4+n,n<P*4,other=0);z+=count[None,:]
    alpha=tl.load(A+h)
    if alpha!=0:
        v=tl.load(V+(h*T+m[:,None])*D+d[None,:],m[:,None]<T,other=0)
        if CENTER:k=k-tl.load(C+h*D+d)[:,None]
        z+=alpha*tl.dot(v,k*k,input_precision='tf32x3')/256
    grouped=tl.reshape(z,(16,16,4));maximum=tl.max(grouped,2)
    score=maximum+tl.log(tl.sum(tl.exp(grouped-maximum[:,:,None]),2))
    pages=tl.program_id(1)*16+tl.arange(0,16)
    tl.store(S+(h*T+m[:,None])*P+pages[None,:],score,(m[:,None]<T)&(pages[None,:]<P))
def route(q,pk,alpha,center=None,logcount=None):
    _,n,h,d=q.shape;t=n//64;p=pk.shape[1]
    mean=torch.empty(h,t,d,device=q.device,dtype=torch.float32);var=torch.empty_like(mean)
    out=torch.empty(h,t,p,device=q.device,dtype=torch.float32)
    _pool[(t,h)](q,mean,var,alpha,h,t,d,True,num_warps=4)
    _score[(triton.cdiv(t,16),triton.cdiv(p,16),h)](mean,var,pk.contiguous(),alpha,mean if center is None else center,
        mean if logcount is None else logcount.contiguous(),out,p,t,d,center is not None,logcount is not None,num_warps=4)
    return out
