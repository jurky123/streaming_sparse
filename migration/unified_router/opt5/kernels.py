"""Recompute QK on chip; never materialize the full logits or weights.

First pass emits page scores and energies. Second pass uses the page scores
as normalizers while directly accumulating two set means. Tensor Core inputs
are FP16, accumulators and scalar reductions FP32.
"""
import math
import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl
from ..opt4.streamgram import prepare_stream


def prepare_flash(prepared):
    state=prepare_stream(prepared)
    keys=[];values=[];counts=[]
    for key,lc,v in prepared[1]:
        h,b,p=lc.shape
        keys.append(F.pad(key.reshape(h,b,p,64),(0,0,0,16-p)).half())
        values.append(F.pad(v,(0,0,0,16-p)).half())
        counts.append(F.pad(lc,(0,16-p),value=-float('inf')))
    state['padded_key']=torch.cat(keys,1)
    state['padded_value']=torch.cat(values,1)
    state['padded_lc']=torch.cat(counts,1)
    state['key16']=state['key'].half()
    state['value16']=state['value'].half()
    return state


@tr.jit
def _score_energy(Q,K,LC,G,V,S,E,N:tl.constexpr,B:tl.constexpr,
                  QM:tl.constexpr,GP:tl.constexpr,CONTENT_ENERGY:tl.constexpr,
                  W=None,START=None,COUNT=None,C=None,R:tl.constexpr=0,KEEP:tl.constexpr=False):
    h=tl.program_id(2);qi=tl.program_id(0)*QM+tl.arange(0,QM)
    pg=tl.program_id(1)*GP+tl.arange(0,GP)
    proto=tl.arange(0,GP*16);pages=tl.program_id(1)*GP+proto//16
    d=tl.arange(0,64);p=tl.arange(0,16)
    q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0.)
    key=tl.load(K+((h*B+pages[None,:])*16+proto[None,:]%16)*64+d[:,None],pages[None,:]<B,0.)
    raw=tl.dot(q,key).reshape((QM,GP,16))
    lc=tl.load(LC+(h*B+pg[:,None])*16+p[None,:],pg[:,None]<B,float('-inf'))
    logits=raw/math.sqrt(128)+lc[None,:,:]
    maximum=tl.max(logits,2);maximum=tl.where(pg[None,:]<B,maximum,0.)
    w=tl.exp(logits-maximum[:,:,None]);den=tl.sum(w,2)
    den=tl.where(pg[None,:]<B,den,1.)
    score=maximum+tl.log(den);w=w/den[:,:,None]
    if KEEP:
        start=tl.load(START+pg,pg<B,0);count=tl.load(COUNT+pg,pg<B,0)
        cost=tl.load(C+pg,pg<B,0.)
        ptr=(h*R+start[None,:,None]+p[None,None,:])*N+qi[:,None,None]
        tl.store(W+ptr,w*cost[None,:,None],(qi[:,None,None]<N)&(pg[None,:,None]<B)&(p[None,None,:]<count[None,:,None]))
    wt=tl.permute(w,(1,0,2))
    if CONTENT_ENERGY:
        vd=tl.arange(0,32)
        value=tl.load(V+((h*B+pg[:,None,None])*16+p[None,:,None])*32+vd[None,None,:],pg[:,None,None]<B,0.)
        content=tl.dot(wt.to(tl.float16),value)
        energy=tl.trans(tl.sum(content*content,2))
    else:
        gram=tl.load(G+(h*B+pg[:,None,None])*256+p[None,:,None]*16+p[None,None,:],pg[:,None,None]<B,0.)
        wg=tl.dot(wt.to(tl.float16),gram.to(tl.float16))
        energy=tl.trans(tl.sum(wg*wt,2))
    tl.store(S+(h*N+qi[:,None])*B+pg[None,:],score,(qi[:,None]<N)&(pg[None,:]<B))
    tl.store(E+(h*N+qi[:,None])*B+pg[None,:],energy,(qi[:,None]<N)&(pg[None,:]<B))


@tr.jit
def _recompute_means(Q,K,V,LC,PAGE,C,S,M,OUT,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr,
                     QM:tl.constexpr,BR:tl.constexpr,SPLITS:tl.constexpr,CHUNK:tl.constexpr):
    h=tl.program_id(2);part=tl.program_id(1)
    qi=tl.program_id(0)*QM+tl.arange(0,QM)
    d=tl.arange(0,64);vr=tl.arange(0,32);rk=tl.arange(0,BR)
    q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0.)
    total=tl.full((QM,32),0.,tl.float32);selected_sum=tl.full((QM,32),0.,tl.float32)
    for off in range(CHUNK//BR):
        r=part*CHUNK+off*BR+rk
        key=tl.load(K+(h*R+r[None,:])*64+d[:,None],r[None,:]<R,0.)
        logits=tl.dot(q,key)
        lc=tl.load(LC+h*R+r,r<R,float('-inf'))
        page=tl.load(PAGE+r,r<R,0)
        score=tl.load(S+(h*N+qi[:,None])*B+page[None,:],(qi[:,None]<N)&(r[None,:]<R),0.)
        cost=tl.load(C+page,r<R,0.)
        weights=tl.exp(logits/math.sqrt(128)+lc[None,:]-score)*cost[None,:]
        selection=tl.load(M+(h*N+qi[:,None])*B+page[None,:],(qi[:,None]<N)&(r[None,:]<R),False)
        value=tl.load(V+(h*R+r[:,None])*32+vr[None,:],r[:,None]<R,0.)
        total=tl.dot(weights.to(tl.float16),value,total)
        selected_sum=tl.dot(tl.where(selection,weights,0.).to(tl.float16),value,selected_sum)
    row=h*N+qi
    tl.store(OUT+(row[:,None]*SPLITS+part)*64+vr[None,:],total,qi[:,None]<N)
    tl.store(OUT+(row[:,None]*SPLITS+part)*64+32+vr[None,:],selected_sum,qi[:,None]<N)


@tr.jit
def _recompute_candidates(Q,K,V,LC,START,COUNT,REM,ADD,CV,N:tl.constexpr,R:tl.constexpr,CS:tl.constexpr):
    row=tl.program_id(0);h=row//N;slot=tl.program_id(1)*CS+tl.arange(0,CS)
    d=tl.arange(0,64);qr=tl.arange(0,16);p=tl.arange(0,CS*16);vd=tl.arange(0,32)
    ri=tl.load(REM+row*8+slot,slot<8,-1);aj=tl.load(ADD+row*8+slot-8,slot>=8,-1)
    page=tl.where(slot<8,ri,aj);valid=page>=0
    start=tl.load(START+page,valid,0);count=tl.load(COUNT+page,valid,0)
    # Expand candidate page starts without storing a gathered key tensor.
    pos=(start[:,None]+tl.arange(0,16)[None,:]).reshape((CS*16,))
    okay=(valid[:,None]&(tl.arange(0,16)[None,:]<count[:,None])).reshape((CS*16,))
    q=tl.load(Q+row*64+d[None,:]+tl.zeros((16,1),tl.int32),(qr[:,None]==0),0.)
    key=tl.load(K+(h*R+pos[None,:])*64+d[:,None],okay[None,:],0.)
    product=tl.dot(q,key)
    logits=tl.sum(tl.where(qr[:,None]==0,product,0.),0).reshape((CS,16))
    lc=tl.load(LC+h*R+pos,okay,float('-inf')).reshape((CS,16))
    logits=logits/math.sqrt(128)+lc
    maximum=tl.where(valid,tl.max(logits,1),0.)
    w=tl.exp(logits-maximum[:,None]);w=w/tl.where(valid,tl.sum(w,1),1.)[:,None]
    value=tl.load(V+(h*R+pos[:,None])*32+vd[None,:],okay[:,None],0.).reshape((CS,16,32))
    content=tl.sum(w[:,:,None]*value,1)
    tl.store(CV+(row*16+slot[:,None])*32+vd[None,:],content)
