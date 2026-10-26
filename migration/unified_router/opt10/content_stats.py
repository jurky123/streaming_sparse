import triton as tr
import triton.language as tl
@tr.jit
def _content_stats(CONTENT,ENERGY,X,M,C,O,ST,REM,ADD,CV,VS,N:tl.constexpr,B:tl.constexpr,BP:tl.constexpr,QUANT:tl.constexpr):
    row=tl.program_id(0);h=row//N;q=row%N
    d=tl.arange(0,32);p=tl.arange(0,BP)
    c=tl.load(C+p,p<B,0.);selected=tl.load(M+row*B+p,p<B,False)
    origins=tl.load(O+p,p<B,0.)
    cost=tl.sum(c,0);sc=c*selected;scost=tl.sum(sc,0)
    values=tl.load(CONTENT+(row*B+p[:,None])*32+d[None,:],p[:,None]<B,0.).to(tl.float32)
    if QUANT:values=values*tl.load(VS+h*B+p,p<B,0.)[:,None]
    total=tl.sum(values*c[:,None],0)/cost
    center=tl.sum(values*sc[:,None],0)/scost
    norm=tl.load(ENERGY+row*B+p,p<B,0.)
    da=tl.maximum(tl.sum(norm*c,0)/cost-tl.sum(total*total,0),0.)
    ds=tl.maximum(tl.sum(norm*sc,0)/scost-tl.sum(center*center,0),0.)
    x=tl.load(X+row*B+p,p<B,0.)
    mean=tl.sum(x,0)/B
    std=tl.maximum(tl.sqrt(tl.sum(tl.where(p<B,(x-mean)*(x-mean),0.),0)/B),1.e-6)
    count=tl.sum(selected.to(tl.int32),0)
    # Single-page variance is exactly zero mathematically.
    ds=tl.where(count==1,0.,ds)
    mz=tl.sum(tl.where(selected,(x-mean)/std,0.),0)/count
    tl.store(ST+row*80+d,total);tl.store(ST+row*80+32+d,center)
    tl.store(ST+row*80+64,ds);tl.store(ST+row*80+65,da)
    tl.store(ST+row*80+66,mean);tl.store(ST+row*80+67,std)
    tl.store(ST+row*80+68,scost);tl.store(ST+row*80+69,cost)
    tl.store(ST+row*80+70,tl.sum(sc*origins,0));tl.store(ST+row*80+71,count)
    tl.store(ST+row*80+72,mz)


    slot=tl.arange(0,16)
    ri=tl.load(REM+row*8+slot,slot<8,-1);aj=tl.load(ADD+row*8+slot-8,slot>=8,-1)
    page=tl.where(slot<8,ri,aj)
    vv=tl.load(CONTENT+(row*B+page[:,None])*32+d[None,:],page[:,None]>=0,0.).to(tl.float32)
    if QUANT:vv=vv*tl.load(VS+h*B+page,page>=0,0.)[:,None]
    tl.store(CV+(row*16+slot[:,None])*32+d[None,:],vv)
