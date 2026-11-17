"""Same Reader arithmetic with scalar coefficients contiguous across pages."""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def feature(U,A,row,head,p,P:tl.constexpr,J:tl.constexpr,PACK_A:tl.constexpr):
    f=tl.full(p.shape,0,tl.float32)
    for i in tl.static_range(8):
        u=tl.load(U+row*8+i)
        if PACK_A:a=tl.load(A+((head*8+i)*4+J)*P+p,p<P,0).to(tl.float32)
        else:a=tl.load(A+head*P*32+p*32+i*4+J,p<P,0).to(tl.float32)
        f=f+u*a
    return lib.tanh(f)


@tr.jit
def scalar_kernel(U,CO,A,Z,LM,MASK,MAP,WORK,LOGSTATE,CAND,T:tl.constexpr,M:tl.constexpr,P:tl.constexpr,K:tl.constexpr,S:tl.constexpr,NS:tl.constexpr,AGG32:tl.constexpr=True,PACK_CO:tl.constexpr=False,PACK_A:tl.constexpr=True):
    dtype:tl.constexpr=tl.float32 if AGG32 else tl.float64
    row=tl.program_id(0);b=row//M;head=b//T
    shard=tl.program_id(1)
    p=shard*S+tl.arange(0,S);d=tl.arange(0,16)
    f0=feature(U,A,row,head,p,P,0,PACK_A);f1=feature(U,A,row,head,p,P,1,PACK_A)
    f2=feature(U,A,row,head,p,P,2,PACK_A);f3=feature(U,A,row,head,p,P,3,PACK_A)
    if PACK_CO:cb=CO+head*P*85+P*80+p
    else:cb=CO+head*P*85+p*85+16
    CB_STEP:tl.constexpr=P if PACK_CO else 17
    RP_STEP:tl.constexpr=16 if PACK_CO else 17
    RP_ROW:tl.constexpr=80 if PACK_CO else 85
    beta=f0*tl.load(cb+CB_STEP*1,p<P,0).to(tl.float32)+f1*tl.load(cb+CB_STEP*2,p<P,0).to(tl.float32)
    beta=beta+f2*tl.load(cb+CB_STEP*3,p<P,0).to(tl.float32);beta=beta+f3*tl.load(cb+CB_STEP*4,p<P,0).to(tl.float32)
    beta=(beta+tl.load(cb,p<P,0).to(tl.float32)).to(dtype)
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
    off=(row*NS+shard)*10*S+tl.arange(0,S)
    tl.store(WORK+off+0*S,f0)
    tl.store(WORK+off+1*S,f1)
    tl.store(WORK+off+2*S,f2)
    tl.store(WORK+off+3*S,f3)
    tl.store(WORK+off+4*S,c0)
    tl.store(WORK+off+5*S,c1)
    tl.store(WORK+off+6*S,c2)
    tl.store(WORK+off+7*S,wc)
    tl.store(WORK+off+8*S,wt)
    tl.store(WORK+off+9*S,wb)
    tl.store(LOGSTATE+(row*NS+shard)*4,mx+lib.log(tl.sum(lib.exp(logs-mx),0)))
    tl.store(LOGSTATE+(row*NS+shard)*4+1,me+lib.log(tl.sum(lib.exp(le-me),0)))
    tl.store(LOGSTATE+(row*NS+shard)*4+2,mb+lib.log(zb))
    tl.store(LOGSTATE+(row*NS+shard)*4+3,mx)
    cp=CAND+(row*K+slot)*34
    valid=(p<P)&(slot>=0)
    tl.store(cp,lm,valid);tl.store(cp+1,beta,valid)

@tr.jit
def vector_kernel(CO,MEAN,DETAIL,MASK,MAP,WORK,STATE,CAND,T:tl.constexpr,M:tl.constexpr,P:tl.constexpr,K:tl.constexpr,S:tl.constexpr,NS:tl.constexpr,VD:tl.constexpr=16,VP:tl.constexpr=False):
    dtype:tl.constexpr=tl.float32
    row=tl.program_id(0);b=row//M;head=b//T;shard=tl.program_id(1)
    p=shard*S+tl.arange(0,S);d=tl.program_id(2)*VD+tl.arange(0,VD)
    RP_STEP:tl.constexpr=16 if VP else 17
    RP_ROW:tl.constexpr=80 if VP else 85
    off=(row*NS+shard)*10*S+tl.arange(0,S)
    f0=tl.load(WORK+off+0*S)
    f1=tl.load(WORK+off+1*S)
    f2=tl.load(WORK+off+2*S)
    f3=tl.load(WORK+off+3*S)
    c0=tl.load(WORK+off+4*S)
    c1=tl.load(WORK+off+5*S)
    c2=tl.load(WORK+off+6*S)
    wc=tl.load(WORK+off+7*S)
    wt=tl.load(WORK+off+8*S)
    wb=tl.load(WORK+off+9*S)
    selected=tl.load(MASK+b*P+p,p<P,0);slot=tl.load(MAP+b*P+p,p<P,-1)
    base=CO+head*P*RP_ROW+p[:,None]*RP_ROW+d[None,:]
    rp=f0[:,None]*tl.load(base+RP_STEP*1,p[:,None]<P,0).to(tl.float32)
    rp=rp+f1[:,None]*tl.load(base+RP_STEP*2,p[:,None]<P,0).to(tl.float32)
    rp=rp+f2[:,None]*tl.load(base+RP_STEP*3,p[:,None]<P,0).to(tl.float32)
    rp=rp+f3[:,None]*tl.load(base+RP_STEP*4,p[:,None]<P,0).to(tl.float32)
    rp=(rp+tl.load(base,p[:,None]<P,0).to(tl.float32)).to(dtype)
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
    cp=CAND+(row*K+slot)*34
    valid=(p<P)&(slot>=0)
    tl.store(cp[:,None]+2+d[None,:],center,valid[:,None])
    tl.store(cp[:,None]+18+d[None,:],rp,valid[:,None])
