import triton as tr
import triton.language as tl
@tr.jit
def _gram_stats(MEAN,SCALE,X,M,C,O,ST,N:tl.constexpr,B:tl.constexpr,BP:tl.constexpr):
    row=tl.program_id(0);h=row//N;q=row%N
    d=tl.arange(0,32);p=tl.arange(0,BP)
    c=tl.load(C+p,p<B,0.);selected=tl.load(M+row*B+p,p<B,False)
    origins=tl.load(O+p,p<B,0.)
    cost=tl.sum(c,0);sc=c*selected;scost=tl.sum(sc,0)
    total=tl.load(MEAN+(h*2*N+q)*64+d)/cost
    center=tl.load(MEAN+(h*2*N+N+q)*64+d)/scost
    scale=tl.load(SCALE+h)
    full_energy=tl.load(MEAN+(h*2*N+q)*64+32)*scale
    selected_energy=tl.load(MEAN+(h*2*N+N+q)*64+32)*scale
    da=tl.maximum(full_energy/cost-tl.sum(total*total,0),0.)
    ds=tl.maximum(selected_energy/scost-tl.sum(center*center,0),0.)
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


