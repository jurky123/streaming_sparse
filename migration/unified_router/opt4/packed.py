"""Pack all prototype groups; one score GEMM and compact Gram statistics."""
import math
import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl


def prepare_packed(prepared):
    basis,parts=prepared
    starts=[];counts=[];offset=0;grams=[]
    for key,lc,v in parts:
        b,p=lc.shape[1:]
        starts.append(torch.arange(b,device=key.device,dtype=torch.int32)*p+offset)
        counts.append(torch.full((b,),p,device=key.device,dtype=torch.int32))
        g=v@v.transpose(-1,-2)
        grams.append(F.pad(g,(0,16-p,0,16-p)))
        offset+=b*p
    return dict(key=torch.cat([k for k,_,_ in parts],1),
                value=torch.cat([v.flatten(1,2)for _,_,v in parts],1),
                lc=torch.cat([lc.flatten(1) for _,lc,_ in parts],1),
                starts=torch.cat(starts),counts=torch.cat(counts),grams=torch.cat(grams,1))


@tr.jit
def _packed_score(L,LC,START,COUNT,S,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr,Q:tl.constexpr):
    q=tl.program_id(0)*Q+tl.arange(0,Q);page=tl.program_id(1);h=tl.program_id(2)
    p=tl.arange(0,16);start=tl.load(START+page);count=tl.load(COUNT+page)
    ll=tl.load(L+(h*N+q[:,None])*R+start+p[None,:],(q[:,None]<N)&(p[None,:]<count),0.)
    lc=tl.load(LC+h*R+start+p,p<count,float('-inf'))
    ll=ll/math.sqrt(128)+lc[None,:]
    maxval=tl.max(ll,1);s=maxval+tl.log(tl.sum(tl.exp(ll-maxval[:,None]),1))
    tl.store(S+(h*N+q)*B+page,s,q<N)


@tr.jit
def _packed_weights(L,LC,G,START,COUNT,M,C,W,E,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr,Q:tl.constexpr,PREC:tl.constexpr):
    q=tl.program_id(0)*Q+tl.arange(0,Q);page=tl.program_id(1);h=tl.program_id(2)
    p=tl.arange(0,16);start=tl.load(START+page);count=tl.load(COUNT+page)
    ll=tl.load(L+(h*N+q[:,None])*R+start+p[None,:],(q[:,None]<N)&(p[None,:]<count),0.)
    lc=tl.load(LC+h*R+start+p,p<count,float('-inf'))
    ll=ll/math.sqrt(128)+lc[None,:]
    w=tl.exp(ll-tl.max(ll,1)[:,None]);w=w/tl.sum(w,1)[:,None]
    gram=tl.load(G+(h*B+page)*256+p[:,None]*16+p[None,:])
    if PREC=='fp16':wg=tl.dot(w.to(tl.float16),gram.to(tl.float16))
    else:wg=tl.dot(w,gram,input_precision=PREC)
    energy=tl.sum(wg*w,1);tl.store(E+(h*N+q)*B+page,energy,q<N)
    c=tl.load(C+page);selected=tl.load(M+(h*N+q)*B+page,q<N,False)
    tl.store(W+(h*2*N+q[:,None])*R+start+p[None,:],w*c,(q[:,None]<N)&(p[None,:]<count))
    tl.store(W+(h*2*N+N+q[:,None])*R+start+p[None,:],w*c*selected[:,None],(q[:,None]<N)&(p[None,:]<count))


@tr.jit
def _packed_candidates(L,LC,V,START,COUNT,REM,ADD,CV,N:tl.constexpr,R:tl.constexpr):
    row=tl.program_id(0);h=row//N
    slot=tl.arange(0,16);p=tl.arange(0,16);d=tl.arange(0,32)
    i=tl.load(REM+row*8+slot,slot<8,-1);j=tl.load(ADD+row*8+slot-8,slot>=8,-1)
    page=tl.where(slot<8,i,j);valid=page>=0
    start=tl.load(START+page,valid,0);count=tl.load(COUNT+page,valid,0)
    ll=tl.load(L+row*R+start[:,None]+p[None,:],valid[:,None]&(p[None,:]<count[:,None]),0.)
    lc=tl.load(LC+h*R+start[:,None]+p[None,:],valid[:,None]&(p[None,:]<count[:,None]),float('-inf'))
    ll=ll/math.sqrt(128)+lc;maxval=tl.where(valid,tl.max(ll,1),0.)
    w=tl.exp(ll-maxval[:,None]);w=w/tl.where(valid,tl.sum(w,1),1.)[:,None]
    v=tl.load(V+(h*R+start[:,None,None]+p[None,:,None])*32+d[None,None,:],
        valid[:,None,None]&(p[None,:,None]<count[:,None,None]),0.)
    content=tl.sum(w[:,:,None]*v,1)
    tl.store(CV+(row*16+slot[:,None])*32+d[None,:],content)
