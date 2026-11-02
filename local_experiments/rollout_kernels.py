"""Fused query moments and page scores; no persistent K-square or V prototypes."""
import torch,triton
import triton.language as tl

@triton.jit
def _pool(Q,M,V,A,H:tl.constexpr,T:tl.constexpr,D:tl.constexpr,USE_VAR:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1)
    r=tl.arange(0,64);d=tl.arange(0,D)
    x=tl.load(Q+((t*64+r[:,None])*H+h)*D+d[None,:]).to(tl.float32)
    mean=tl.sum(x,0)/64
    tl.store(M+(h*T+t)*D+d,mean)
    if USE_VAR:
        alpha=tl.load(A+h)
        if alpha!=0:
            var=tl.maximum(tl.sum(x*x,0)/64-mean*mean,0.)
        else:var=tl.full((D,),0.,tl.float32)
        tl.store(V+(h*T+t)*D+d,var)

@triton.jit
def _score(M,V,K,A,S,P:tl.constexpr,T:tl.constexpr,D:tl.constexpr,USE_VAR:tl.constexpr):
    m=tl.program_id(0)*16+tl.arange(0,16)
    n=tl.program_id(1)*64+tl.arange(0,64)
    h=tl.program_id(2);d=tl.arange(0,D)
    q=tl.load(M+(h*T+m[:,None])*D+d[None,:],m[:,None]<T,other=0)
    k=tl.load(K+(h*P*4+n[None,:])*D+d[:,None],n[None,:]<P*4,other=0)
    z=tl.dot(q,k,input_precision='tf32x3')*0.08838834764831845
    if USE_VAR:
        alpha=tl.load(A+h)
        if alpha!=0:
            v=tl.load(V+(h*T+m[:,None])*D+d[None,:],m[:,None]<T,other=0)
            z+=alpha*tl.dot(v,k*k,input_precision='tf32x3')/256
    grouped=tl.reshape(z,(16,16,4));maximum=tl.max(grouped,2)
    score=maximum+tl.log(tl.sum(tl.exp(grouped-maximum[:,:,None]),2))
    pages=tl.program_id(1)*16+tl.arange(0,16)
    tl.store(S+(h*T+m[:,None])*P+pages[None,:],score,(m[:,None]<T)&(pages[None,:]<P))

@triton.jit
def _aggregate(K,W,O,H:tl.constexpr,P:tl.constexpr,D:tl.constexpr,LEARNED:tl.constexpr):
    p=tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,D)
    k=tl.load(K+((h*P+p)*64+t[:,None])*D+d[None,:]).to(tl.float32)
    kg=tl.reshape(k,(4,16,D))
    if LEARNED:
        w=tl.load(W+(h*4+t[:,None]//16)*D+d[None,:])
        dots=tl.reshape(tl.sum(k*w,1),(4,16));center=tl.sum(dots,1)/16
        u=(dots-center[:,None])/2
        bounded=2*(2*tl.sigmoid(2*u)-1)
        prob=tl.exp(bounded-tl.max(bounded,1)[:,None]);prob=prob/tl.sum(prob,1)[:,None]
        out=tl.sum(kg*prob[:,:,None],1)
    else:out=tl.sum(kg,1)/16
    r=tl.arange(0,4)
    tl.store(O+((h*P+p)*4+r[:,None])*D+d[None,:],out)

def aggregate(k,weight=None):
    h,n,d=k.shape;p=n//64;out=torch.empty(h,p,4,d,device=k.device,dtype=torch.float32)
    _aggregate[(p,h)](k,k if weight is None else weight,out,h,p,d,weight is not None,num_warps=4)
    return out

def route_scores(q,pk,alpha=None):
    _,n,h,d=q.shape;t=n//64;p=pk.shape[1]
    mean=torch.empty(h,t,d,device=q.device,dtype=torch.float32);var=torch.empty_like(mean)
    out=torch.empty(h,t,p,device=q.device,dtype=torch.float32)
    _pool[(t,h)](q,mean,var,mean if alpha is None else alpha,h,t,d,alpha is not None,num_warps=4)
    # Prefix of a fixed-capacity bank is not contiguous across heads; use stride below via contiguous pack.
    keys=pk.contiguous()
    _score[(triton.cdiv(t,16),triton.cdiv(p,16),h)](mean,var,keys,mean if alpha is None else alpha,out,p,t,d,alpha is not None,num_warps=4)
    return out
