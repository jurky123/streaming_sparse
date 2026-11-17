"""Delay P×16 vectors until scalar weights are ready to reduce live registers."""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .reader_reduce import feature

@tr.jit
def reader_kernel(U,CO,A,MEAN,DETAIL,Z,LM,MASK,MAP,STATE,LOGSTATE,CAND,
                  T:tl.constexpr,M:tl.constexpr,P:tl.constexpr,K:tl.constexpr,S:tl.constexpr,NS:tl.constexpr,AGG32:tl.constexpr):
    dtype:tl.constexpr=tl.float32 if AGG32 else tl.float64
    row=tl.program_id(0);b=row//M;head=b//T
    shard=tl.program_id(1)
    p=shard*S+tl.arange(0,S);d=tl.arange(0,16)
    f0=feature(U,A,row,head,p,P,0);f1=feature(U,A,row,head,p,P,1)
    f2=feature(U,A,row,head,p,P,2);f3=feature(U,A,row,head,p,P,3)
    cb=CO+head*P*85+p*85+16
    beta=f0*tl.load(cb+17,p<P,0)+f1*tl.load(cb+34,p<P,0)
    beta=beta+f2*tl.load(cb+51,p<P,0);beta=beta+f3*tl.load(cb+68,p<P,0)
    beta=(beta+tl.load(cb,p<P,0)).to(dtype)
    z0=tl.load(Z+(row*P+p)*4,p<P,0).to(dtype)
    z1=tl.load(Z+(row*P+p)*4+1,p<P,0).to(dtype)
    z2=tl.load(Z+(row*P+p)*4+2,p<P,0).to(dtype)
    z3=tl.load(Z+(row*P+p)*4+3,p<P,0).to(dtype)
    zm=tl.maximum(tl.maximum(z0,z1),tl.maximum(z2,z3))
    w0=lib.exp(z0-zm);w1=lib.exp(z1-zm);w2=lib.exp(z2-zm);w3=lib.exp(z3-zm)
    total=w0+w1+w2+w3;w0=w0/total;w1=w1/total;w2=w2/total;w3=w3/total
    c0=w0+w1-w2-w3;c1=w0-w1+w2-w3;c2=w0-w1-w2+w3
    lm=tl.load(LM+row*P+p,p<P,-float('inf')).to(dtype)
    # All mixtures and endpoints share this translation; it cancels in deltas.
    selected=tl.load(MASK+b*P+p,p<P,0);slot=tl.load(MAP+b*P+p,p<P,-1)
    le=lm+beta;logs=tl.where(selected,le,lm)
    mx=tl.max(logs,0);me=tl.max(le,0);bg=tl.where(slot>=0,-float('inf'),logs);mb=tl.max(bg,0)
    wc=lib.exp(logs-mx);wc=wc/tl.sum(wc,0)
    wt=lib.exp(le-me);wt=wt/tl.sum(wt,0)
    wb=lib.exp(bg-tl.where(mb>-float('inf'),mb,0.));zb=tl.sum(wb,0);wb=wb/tl.where(zb>0,zb,1.)
    base=CO+head*P*85+p[:,None]*85+d[None,:]
    rp=f0[:,None]*tl.load(base+17,p[:,None]<P,0)
    rp=rp+f1[:,None]*tl.load(base+34,p[:,None]<P,0)
    rp=rp+f2[:,None]*tl.load(base+51,p[:,None]<P,0)
    rp=rp+f3[:,None]*tl.load(base+68,p[:,None]<P,0)
    rp=(rp+tl.load(base,p[:,None]<P,0)).to(dtype)
    db=DETAIL+head*P*48+p[:,None]*48+d[None,:]
    approx=c0[:,None]*tl.load(db,p[:,None]<P,0).to(dtype)
    approx=approx+c1[:,None]*tl.load(db+16,p[:,None]<P,0).to(dtype)
    approx=approx+c2[:,None]*tl.load(db+32,p[:,None]<P,0).to(dtype)
    approx=approx+tl.load(MEAN+head*P*16+p[:,None]*16+d[None,:],p[:,None]<P,0).to(dtype)
    center=approx
    current=center+tl.where(selected[:,None],rp,0.)
    h=tl.sum(wc[:,None]*current,0);target=tl.sum(wt[:,None]*(center+rp),0)
    ob=tl.sum(wb[:,None]*current,0)
    tl.store(STATE+(row*NS+shard)*48+d,h);tl.store(STATE+(row*NS+shard)*48+16+d,target);tl.store(STATE+(row*NS+shard)*48+32+d,ob)
    tl.store(LOGSTATE+(row*NS+shard)*4,mx+lib.log(tl.sum(lib.exp(logs-mx),0)))
    tl.store(LOGSTATE+(row*NS+shard)*4+1,me+lib.log(tl.sum(lib.exp(le-me),0)))
    tl.store(LOGSTATE+(row*NS+shard)*4+2,mb+lib.log(zb))
    tl.store(LOGSTATE+(row*NS+shard)*4+3,mx)
    cp=CAND+(row*K+slot)*34
    valid=(p<P)&(slot>=0)
    tl.store(cp,lm,valid);tl.store(cp+1,beta,valid)
    tl.store(cp[:,None]+2+d[None,:],center,valid[:,None])
    tl.store(cp[:,None]+18+d[None,:],rp,valid[:,None])
