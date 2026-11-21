"""Probe-parallel Reader. Per-shard reductions are contracted in tensor cores."""
import triton
import triton.language as tl

@triton.jit
def vector_state(CO,MEAN,DETAIL,MASK,WORK,STATE,T:tl.constexpr,P:tl.constexpr,S:tl.constexpr,NS:tl.constexpr):
    b=tl.program_id(0)
    sh=tl.program_id(1)
    h=b//T
    m=tl.arange(0,16)
    i=tl.arange(0,S)
    p=sh*S+i
    d=tl.arange(0,16)
    work=WORK+((b*16+m[:,None])*NS+sh)*10*S+i[None,:]
    wc=tl.load(work+7*S)
    wt=tl.load(work+8*S)
    wb=tl.load(work+9*S)
    selected=tl.load(MASK+b*P+p,p<P,0)
    hsum=tl.full((16,16),0.,tl.float32)
    target=tl.full((16,16),0.,tl.float32)
    bg=tl.full((16,16),0.,tl.float32)
    for j in tl.static_range(9):
        if j<4:
            if j==0:
                coeff=tl.full((16,S),1.,tl.float32)
                base=MEAN+(h*P+p[:,None])*16+d[None,:]
            else:
                coeff=tl.load(work+(j+3)*S)
                base=DETAIL+(h*P+p[:,None])*48+(j-1)*16+d[None,:]
            weight=tl.load(base,p[:,None]<P,0).to(tl.float16)
            hc=wc*coeff
            bc=wb*coeff
        else:
            if j==4: coeff=tl.full((16,S),1.,tl.float32)
            else: coeff=tl.load(work+(j-5)*S)
            base=CO+(h*P+p[:,None])*85+(j-4)*17+d[None,:]
            weight=tl.load(base,p[:,None]<P,0).to(tl.float16)
            hc=tl.where(selected[None,:],wc*coeff,0.)
            bc=tl.where(selected[None,:],wb*coeff,0.)
        hsum=tl.dot(hc.to(tl.float16),weight,hsum)
        target=tl.dot((wt*coeff).to(tl.float16),weight,target)
        bg=tl.dot(bc.to(tl.float16),weight,bg)
    dst=STATE+((b*16+m[:,None])*NS+sh)*48+d[None,:]
    tl.store(dst,hsum);tl.store(dst+16,target);tl.store(dst+32,bg)

@triton.jit
def endpoints(CO,MEAN,DETAIL,WORK,IDS,CAND,T:tl.constexpr,P:tl.constexpr,S:tl.constexpr,NS:tl.constexpr,K:tl.constexpr):
    row=tl.program_id(0);b=row//16;h=b//T
    j=tl.program_id(1)*16+tl.arange(0,16)
    d=tl.arange(0,16)
    p=tl.load(IDS+b*K+j,j<K,0)
    w=WORK+(row*NS+p//S)*10*S+p%S
    base=CO+(h*P+p[:,None])*85+d[None,:]
    rp=tl.load(w)[:,None]*tl.load(base+17).to(tl.float32)
    for n in tl.static_range(1,4):
        rp=rp+tl.load(w+n*S)[:,None]*tl.load(base+(n+1)*17).to(tl.float32)
    rp+=tl.load(base).to(tl.float32)
    base=DETAIL+(h*P+p[:,None])*48+d[None,:]
    center=tl.load(w+4*S)[:,None]*tl.load(base).to(tl.float32)
    for n in tl.static_range(1,3):
        center=center+tl.load(w+(n+4)*S)[:,None]*tl.load(base+n*16).to(tl.float32)
    center+=tl.load(MEAN+(h*P+p[:,None])*16+d[None,:]).to(tl.float32)
    dest=CAND+(row*K+j[:,None])*34+d[None,:]
    tl.store(dest+2,center,j[:,None]<K);tl.store(dest+18,rp,j[:,None]<K)
