"""Local precision-controlled GEMM and compact routing outputs.

The global PyTorch TF32 flag remains disabled; precision changes are confined
to the explicit Triton dot calls below and the routing kernels that use them.
"""
import math
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _mm(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        AH: tl.constexpr, AM: tl.constexpr, AK: tl.constexpr,
        BH: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
        BM: tl.constexpr, BN_: tl.constexpr, BK_: tl.constexpr, PREC: tl.constexpr):
    h = tl.program_id(2)
    m = tl.program_id(0)*BM + tl.arange(0,BM)
    n = tl.program_id(1)*BN_ + tl.arange(0,BN_)
    kk = tl.arange(0,BK_)
    acc = tl.full((BM,BN_),0.,tl.float32)
    for off in range(tr.cdiv(K,BK_)):
        k = off*BK_+kk
        a = tl.load(A+h*AH+m[:,None]*AM+k[None,:]*AK,(m[:,None]<M)&(k[None,:]<K),0.)
        b = tl.load(B+h*BH+k[:,None]*BK+n[None,:]*BN,(k[:,None]<K)&(n[None,:]<N),0.)
        if PREC == 'fp16':
            acc = tl.dot(a.to(tl.float16),b.to(tl.float16),acc)
        else:
            acc = tl.dot(a,b,acc,input_precision=PREC)
    tl.store(Y+(h*M+m[:,None])*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))


def mm(a,b,precision='tf32x3'):
    h,m,k=a.shape
    n=b.shape[2]
    y=torch.empty((h,m,n),device=a.device,dtype=torch.float32)
    _mm[(tr.cdiv(m,32),tr.cdiv(n,64),h)](a,b,y,m,n,k,*a.stride(),*b.stride(),32,64,32,precision,
        num_warps=4,num_stages=2)
    return y


@tr.jit
def _weights_energy(L, LC, G, MASK, COST, W, E, N:tl.constexpr, B:tl.constexpr,
        P:tl.constexpr,TOTAL:tl.constexpr,OFFSET:tl.constexpr,R:tl.constexpr,ROFF:tl.constexpr,
        Q:tl.constexpr,PP:tl.constexpr,PREC:tl.constexpr):
    q=tl.program_id(0)*Q+tl.arange(0,Q)
    page=tl.program_id(1);h=tl.program_id(2);p=tl.arange(0,PP)
    logits=tl.load(L+((h*N+q[:,None])*B+page)*P+p[None,:],(q[:,None]<N)&(p[None,:]<P),0.)
    lc=tl.load(LC+(h*B+page)*P+p,p<P,float('-inf'))
    logits=logits/math.sqrt(128)+lc[None,:]
    w=tl.exp(logits-tl.max(logits,1)[:,None]);w=w/tl.sum(w,1)[:,None]
    gram=tl.load(G+((h*B+page)*P+p[:,None])*P+p[None,:],(p[:,None]<P)&(p[None,:]<P),0.)
    if PREC == 'fp16':
        wg=tl.dot(w.to(tl.float16),gram.to(tl.float16))
    else:
        wg=tl.dot(w,gram,input_precision=PREC)
    energy=tl.sum(wg*w,1)
    tl.store(E+(h*N+q)*TOTAL+OFFSET+page,energy,q<N)
    c=tl.load(COST+OFFSET+page)
    selected=tl.load(MASK+(h*N+q)*TOTAL+OFFSET+page,q<N,False)
    tl.store(W+((h*2*N+q[:,None])*R+ROFF+page*P+p[None,:]),w*c,(q[:,None]<N)&(p[None,:]<P))
    tl.store(W+((h*2*N+N+q[:,None])*R+ROFF+page*P+p[None,:]),w*c*selected[:,None],(q[:,None]<N)&(p[None,:]<P))


@tr.jit
def _gram_stats(MEAN,ENERGY,X,M,C,O,ST,N:tl.constexpr,B:tl.constexpr,BP:tl.constexpr):
    row=tl.program_id(0);h=row//N;q=row%N
    d=tl.arange(0,32);p=tl.arange(0,BP)
    c=tl.load(C+p,p<B,0.);selected=tl.load(M+row*B+p,p<B,False)
    origins=tl.load(O+p,p<B,0.)
    cost=tl.sum(c,0);sc=c*selected;scost=tl.sum(sc,0)
    total=tl.load(MEAN+(h*2*N+q)*32+d)/cost
    center=tl.load(MEAN+(h*2*N+N+q)*32+d)/scost
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


@tr.jit
def _candidate_content(L,LC,V,REM,ADD,OUT,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,
                       OFFSET:tl.constexpr,PP:tl.constexpr):
    row=tl.program_id(0);h=row//N;q=row%N
    slot=tl.arange(0,16);p=tl.arange(0,PP);d=tl.arange(0,32)
    ri=tl.load(REM+row*8+slot,slot<8,-1)
    aj=tl.load(ADD+row*8+slot-8,slot>=8,-1)
    page=tl.where(slot<8,ri,aj)-OFFSET
    valid=(page>=0)&(page<B)
    ll=tl.load(L+(row*B+page[:,None])*P+p[None,:],valid[:,None]&(p[None,:]<P),0.)
    lc=tl.load(LC+(h*B+page[:,None])*P+p[None,:],valid[:,None]&(p[None,:]<P),float('-inf'))
    ll=ll/math.sqrt(128)+lc
    maxval=tl.where(valid,tl.max(ll,1),0.)
    w=tl.exp(ll-maxval[:,None]);w=w/tl.where(valid,tl.sum(w,1),1.)[:,None]
    value=tl.load(V+((h*B+page[:,None,None])*P+p[None,:,None])*32+d[None,None,:],
                  valid[:,None,None]&(p[None,:,None]<P),0.)
    content=tl.sum(w[:,:,None]*value,1)
    tl.store(OUT+(row*16+slot[:,None])*32+d[None,:],content,valid[:,None])
