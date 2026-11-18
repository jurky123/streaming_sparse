"""Factor each page's Risk contribution by the shared mass of its probe row."""
import math
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _partial(Q,K,U,QN,ST,TH,C,VAL,NORM,T:tl.constexpr,P:tl.constexpr,BP:tl.constexpr,BK:tl.constexpr,FAST_SQRT:tl.constexpr):
    b=tl.program_id(0)
    block=tl.program_id(1)
    head=b//T
    m=tl.arange(0,16)
    n=block*(4*BP)+tl.arange(0,4*BP)
    p=block*BP+tl.arange(0,BP)
    d=tl.arange(0,BK)
    rep=tl.arange(0,4)
    acc=tl.full((16,4*BP),0,tl.float32)
    for j in tl.static_range(128//BK):
        dim=j*BK+d
        q=tl.load(Q+(b*16+m[:,None])*128+dim[None,:])
        k=tl.load(K+(head*P*4+n[None,:])*128+dim[:,None],n[None,:]<P*4,0)
        acc=tl.dot(q,k,acc)
    logits=tl.where(n[None,:]<P*4,acc*(128**-.5),-float('inf'))
    mx=tl.max(logits,1)
    mass=tl.exp(logits-mx[:,None])
    den=tl.sum(mass,1)
    w=tl.reshape(mass,(16,BP,4))
    st=ST+((head*P+p[None,:,None])*4+rep[None,None,:])*12
    u=tl.full((16,BP,4),0,tl.float32)
    for g in tl.static_range(8):
        q2=tl.load(U+(b*16+m)*8+g)
        var=tl.load(st+g,p[None,:,None]<P,0).to(tl.float32)
        u=tl.fma(q2[:,None,None],var,u)
    u=tl.maximum(u,0.)
    beta=tl.load(QN+b*16+m)[:,None,None]*tl.load(st+8,p[None,:,None]<P,0).to(tl.float32)
    alpha=tl.sum(w,2)
    mu=tl.load(ST+((head*P+p)*4)*12+11,p<P,0).to(tl.float32)
    c=tl.load(C+head*P+p,p<P,0)
    x0=alpha*mu[None,:]+tl.sum(tl.abs(w-alpha[:,:,None]*.25)*tl.load(st+9,p[None,:,None]<P,0).to(tl.float32),2)
    root=tl.sqrt(u) if FAST_SQRT else libdevice.sqrt(u)
    x1=tl.sum(w*root*tl.load(st+10,p[None,:,None]<P,0).to(tl.float32),2)
    x2=c[None,:]*tl.sum(w*beta,2)
    x3=c[None,:]*tl.sum(w*(u+beta*beta),2)
    val=((x0*tl.load(TH+head*4)+x1*tl.load(TH+head*4+1))+x2*tl.load(TH+head*4+2))+x3*tl.load(TH+head*4+3)
    row=(b*triton.cdiv(P,BP)+block)*16+m
    tl.store(VAL+row[:,None]*BP+tl.arange(0,BP)[None,:],val)
    tl.store(NORM+row*2,mx)
    tl.store(NORM+row*2+1,den)


