"""Share page coefficient loads over a small group of probe queries (FP32)."""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

@tr.jit
def feature(U,A,rows,head,p,P:tl.constexpr,J:tl.constexpr,G:tl.constexpr,S:tl.constexpr):
    f=tl.full((G,S),0,tl.float32)
    for i in tl.static_range(8):
        u=tl.load(U+rows*8+i);a=tl.load(A+head*P*32+p*32+i*4+J,p<P,0)
        f=f+u[:,None]*a[None,:]
    return lib.tanh(f)

@tr.jit
def grouped_reader_kernel(U,CO,A,MEAN,DETAIL,Z,LM,MASK,MAP,STATE,LOGSTATE,CAND,
                          T:tl.constexpr,M:tl.constexpr,P:tl.constexpr,K:tl.constexpr,S:tl.constexpr,NS:tl.constexpr,G:tl.constexpr):
    group=tl.program_id(0);shard=tl.program_id(1);b=group//(M//G);head=b//T
    rows=b*M+(group%(M//G))*G+tl.arange(0,G);p=shard*S+tl.arange(0,S);d=tl.arange(0,16)
    f0=feature(U,A,rows,head,p,P,0,G,S);f1=feature(U,A,rows,head,p,P,1,G,S)
    f2=feature(U,A,rows,head,p,P,2,G,S);f3=feature(U,A,rows,head,p,P,3,G,S)
    cb=CO+head*P*85+p[:,None]*85+d[None,:]
    rp=f0[:,:,None]*tl.load(cb+17,p[:,None]<P,0)[None,:,:]
    rp=rp+f1[:,:,None]*tl.load(cb+34,p[:,None]<P,0)[None,:,:]
    rp=rp+f2[:,:,None]*tl.load(cb+51,p[:,None]<P,0)[None,:,:]
    rp=rp+f3[:,:,None]*tl.load(cb+68,p[:,None]<P,0)[None,:,:]
    rp=rp+tl.load(cb,p[:,None]<P,0)[None,:,:]
    cb=CO+head*P*85+p*85+16
    beta=f0*tl.load(cb+17,p<P,0)[None,:]+f1*tl.load(cb+34,p<P,0)[None,:]
    beta=beta+f2*tl.load(cb+51,p<P,0)[None,:];beta=beta+f3*tl.load(cb+68,p<P,0)[None,:]
    beta=beta+tl.load(cb,p<P,0)[None,:]
    zp=(rows[:,None]*P+p[None,:])*4
    z0=tl.load(Z+zp,p[None,:]<P,0);z1=tl.load(Z+zp+1,p[None,:]<P,0)
    z2=tl.load(Z+zp+2,p[None,:]<P,0);z3=tl.load(Z+zp+3,p[None,:]<P,0)
    zm=tl.maximum(tl.maximum(z0,z1),tl.maximum(z2,z3))
    w0=lib.exp(z0-zm);w1=lib.exp(z1-zm);w2=lib.exp(z2-zm);w3=lib.exp(z3-zm)
    total=w0+w1+w2+w3;w0=w0/total;w1=w1/total;w2=w2/total;w3=w3/total
    c0=w0+w1-w2-w3;c1=w0-w1+w2-w3;c2=w0-w1-w2+w3
    db=DETAIL+head*P*48+p[:,None]*48+d[None,:]
    center=c0[:,:,None]*tl.load(db,p[:,None]<P,0).to(tl.float32)[None,:,:]
    center=center+c1[:,:,None]*tl.load(db+16,p[:,None]<P,0).to(tl.float32)[None,:,:]
    center=center+c2[:,:,None]*tl.load(db+32,p[:,None]<P,0).to(tl.float32)[None,:,:]
    center=center+tl.load(MEAN+head*P*16+p[:,None]*16+d[None,:],p[:,None]<P,0).to(tl.float32)[None,:,:]
    lm=tl.load(LM+rows[:,None]*P+p[None,:],p[None,:]<P,-float('inf')).to(tl.float32)
    selected=tl.load(MASK+b*P+p,p<P,0);slot=tl.load(MAP+b*P+p,p<P,-1)
    le=lm+beta;logs=tl.where(selected[None,:],le,lm)
    mx=tl.max(logs,1);me=tl.max(le,1);bg=tl.where(slot[None,:]>=0,-float('inf'),logs);mb=tl.max(bg,1)
    wc=lib.exp(logs-mx[:,None]);zc=tl.sum(wc,1);wc=wc/zc[:,None]
    wt=lib.exp(le-me[:,None]);zt=tl.sum(wt,1);wt=wt/zt[:,None]
    wb=lib.exp(bg-tl.where(mb>-float('inf'),mb,0.)[:,None]);zb=tl.sum(wb,1);wb=wb/tl.where(zb>0,zb,1.)[:,None]
    current=center+tl.where(selected[None,:,None],rp,0.)
    h=tl.sum(wc[:,:,None]*current,1);target=tl.sum(wt[:,:,None]*(center+rp),1);ob=tl.sum(wb[:,:,None]*current,1)
    dest=STATE+(rows[:,None]*NS+shard)*48+d[None,:]
    tl.store(dest,h);tl.store(dest+16,target);tl.store(dest+32,ob)
    dest=LOGSTATE+(rows*NS+shard)*4
    tl.store(dest,mx+lib.log(zc));tl.store(dest+1,me+lib.log(zt));tl.store(dest+2,mb+lib.log(zb));tl.store(dest+3,mx)
    cp=CAND+(rows[:,None]*K+slot[None,:])*34;valid=(p<P)&(slot>=0)
    tl.store(cp,lm,valid[None,:]);tl.store(cp+1,beta,valid[None,:])
    tl.store(cp[:,:,None]+2+d[None,None,:],center,valid[None,:,None]);tl.store(cp[:,:,None]+18+d[None,None,:],rp,valid[None,:,None])
