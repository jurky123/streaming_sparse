"""Experimental clean-commit fusion. Frozen acceptance implementation is untouched."""
import torch,triton
import triton.language as tl

@triton.jit
def _commit(K,V,ORDER,W,CK,CV,PK,WRITE,CAP:tl.constexpr,H:tl.constexpr,D:tl.constexpr,LEARNED:tl.constexpr):
    p=tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,D)
    ids=tl.load(ORDER+(h*72+p)*64+t)
    src=(ids[:,None]*H+h)*D+d[None,:]
    k=tl.load(K+src);v=tl.load(V+src)
    slot=(WRITE+p)%CAP;dst=((slot*64+t[:,None])*H+h)*D+d[None,:]
    tl.store(CK+dst,k);tl.store(CV+dst,v)
    k=k.to(tl.float32);kg=tl.reshape(k,(4,16,D))
    if LEARNED:
        w=tl.load(W+(h*4+t[:,None]//16)*D+d[None,:])
        dots=tl.reshape(tl.sum(k*w,1),(4,16));center=tl.sum(dots,1)/16
        u=(dots-center[:,None])/2;bounded=2*(2*tl.sigmoid(2*u)-1)
        prob=tl.exp(bounded-tl.max(bounded,1)[:,None]);prob=prob/tl.sum(prob,1)[:,None]
        out=tl.sum(kg*prob[:,:,None],1)
    else:out=tl.sum(kg,1)/16
    r=tl.arange(0,4)
    tl.store(PK+((h*CAP+slot)*4+r[:,None])*D+d[None,:],out)

def commit(k,v,order,weight,cache,pk,write,cap):
    _commit[(72,12)](k,v,order,k if weight is None else weight,cache['k'],cache['v'],pk,write,cap,12,128,weight is not None,num_warps=4)
