"""Recompute just the 16 candidate pages without zero-padded MMA rows."""
import math
import triton as tr
import triton.language as tl

@tr.jit
def candidate_simt(Q,K,V,LC,START,COUNT,REM,ADD,CV,N:tl.constexpr,R:tl.constexpr,CS:tl.constexpr):
    row=tl.program_id(0);h=row//N;slot=tl.program_id(1)*CS+tl.arange(0,CS)
    d=tl.arange(0,64);p=tl.arange(0,16);vd=tl.arange(0,32)
    ri=tl.load(REM+row*8+slot,slot<8,-1);aj=tl.load(ADD+row*8+slot-8,slot>=8,-1)
    page=tl.where(slot<8,ri,aj);valid=page>=0
    start=tl.load(START+page,valid,0);count=tl.load(COUNT+page,valid,0)
    pos=start[:,None]+p[None,:];okay=valid[:,None]&(p[None,:]<count[:,None])
    q=tl.load(Q+row*64+d).to(tl.float32)
    key=tl.load(K+(h*R+pos[:,:,None])*64+d[None,None,:],okay[:,:,None],0.).to(tl.float32)
    raw=tl.sum(key*q[None,None,:],2)
    lc=tl.load(LC+h*R+pos,okay,float('-inf'))
    logits=raw/math.sqrt(128)+lc
    maximum=tl.where(valid,tl.max(logits,1),0.)
    w=tl.exp(logits-maximum[:,None]);w=w/tl.where(valid,tl.sum(w,1),1.)[:,None]
    value=tl.load(V+(h*R+pos[:,:,None])*32+vd[None,None,:],okay[:,:,None],0.)
    content=tl.sum(w[:,:,None]*value,1)
    tl.store(CV+(row*16+slot[:,None])*32+vd[None,:],content)
