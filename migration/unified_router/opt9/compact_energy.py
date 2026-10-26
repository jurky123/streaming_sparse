"""Compute real P2/P4/P8/P16 QK; pad only the on-chip PV operand."""
import math
import torch,triton as tr
import triton.language as tl
from ..opt8.projection import project
@tr.jit
def _part(Q,K,LC,V,S,E,W,C,N:tl.constexpr,B:tl.constexpr,NP:tl.constexpr,OFF:tl.constexpr,
          RP:tl.constexpr,R:tl.constexpr,P:tl.constexpr,QM:tl.constexpr,GP:tl.constexpr):
 h=tl.program_id(2);qi=tl.program_id(0)*QM+tl.arange(0,QM)
 pg=tl.program_id(1)*GP+tl.arange(0,GP);proto=tl.arange(0,GP*P)
 pages=tl.program_id(1)*GP+proto//P;d=tl.arange(0,64);p=tl.arange(0,P)
 q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0.)
 key=tl.load(K+((h*B+OFF+pages[None,:])*16+proto[None,:]%P)*64+d[:,None],pages[None,:]<NP,0.)
 raw=tl.dot(q,key).reshape((QM,GP,P))
 lc=tl.load(LC+(h*B+OFF+pg[:,None])*16+p[None,:],pg[:,None]<NP,float('-inf'))
 logits=raw/math.sqrt(128)+lc[None,:,:]
 maximum=tl.max(logits,2);maximum=tl.where(pg[None,:]<NP,maximum,0.)
 w=tl.exp(logits-maximum[:,:,None]);den=tl.sum(w,2);den=tl.where(pg[None,:]<NP,den,1.)
 score=maximum+tl.log(den);w=w/den[:,:,None]
 cost=tl.load(C+OFF+pg,pg<NP,0.)
 ptr=(h*R+RP+pg[None,:,None]*P+p[None,None,:])*N+qi[:,None,None]
 tl.store(W+ptr,w*cost[None,:,None],(qi[:,None,None]<N)&(pg[None,:,None]<NP))
 if P==2:w=tl.permute(tl.join(w,tl.full((QM,GP,2),0.,tl.float32)),(0,1,3,2)).reshape((QM,GP,4))
 if P<=4:w=tl.permute(tl.join(w,tl.full((QM,GP,4),0.,tl.float32)),(0,1,3,2)).reshape((QM,GP,8))
 if P<=8:w=tl.permute(tl.join(w,tl.full((QM,GP,8),0.,tl.float32)),(0,1,3,2)).reshape((QM,GP,16))
 pp=tl.arange(0,16);vd=tl.arange(0,32)
 value=tl.load(V+((h*B+OFF+pg[:,None,None])*16+pp[None,:,None])*32+vd[None,None,:],pg[:,None,None]<NP,0.)
 content=tl.dot(tl.permute(w,(1,0,2)).to(tl.float16),value)
 energy=tl.trans(tl.sum(content*content,2))
 tl.store(S+(h*N+qi[:,None])*B+OFF+pg[None,:],score,(qi[:,None]<N)&(pg[None,:]<NP))
 tl.store(E+(h*N+qi[:,None])*B+OFF+pg[None,:],energy,(qi[:,None]<N)&(pg[None,:]<NP))

def score(q,state,qm=32,width=32):
 h,n,_=q.shape;b=state['costs'].numel();r=state['value'].shape[1];qq=project(q,state['prepared'][0])
 score=torch.empty((h,n,b),device=q.device,dtype=torch.float32);energy=torch.empty_like(score);w=torch.empty((h,r,n),device=q.device,dtype=torch.float16)
 off=rp=0
 for _,lc,_ in state['prepared'][1]:
  _,np,p=lc.shape;gp=max(4,width//p)
  _part[(tr.cdiv(n,qm),tr.cdiv(np,gp),h)](qq,state['padded_key'],state['padded_lc'],state['padded_value'],score,energy,w,state['costs'],n,b,np,off,rp,r,p,qm,gp,num_warps=4,num_stages=1)
  off+=np;rp+=np*p
 return score,(w,energy)
