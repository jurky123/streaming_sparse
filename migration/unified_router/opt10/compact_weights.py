"""Real-P QK/softmax weights; no padded per-page PV energy computation."""
import math
import torch,triton as tr
import triton.language as tl
from ..opt8.projection import project
@tr.jit
def _part(Q,K,LC,S,W,C,N:tl.constexpr,B:tl.constexpr,TOTAL:tl.constexpr,OFFSET:tl.constexpr,
          P:tl.constexpr,QM:tl.constexpr,GP:tl.constexpr,R:tl.constexpr,ROFF:tl.constexpr):
 h=tl.program_id(2);qi=tl.program_id(0)*QM+tl.arange(0,QM);pg=tl.program_id(1)*GP+tl.arange(0,GP)
 proto=tl.arange(0,GP*P);pages=tl.program_id(1)*GP+proto//P;d=tl.arange(0,64);p=tl.arange(0,P)
 q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0.)
 key=tl.load(K+((h*TOTAL+OFFSET+pages[None,:])*16+proto[None,:]%P)*64+d[:,None],pages[None,:]<B,0.)
 raw=tl.dot(q,key).reshape((QM,GP,P))
 lc=tl.load(LC+(h*TOTAL+OFFSET+pg[:,None])*16+p[None,:],pg[:,None]<B,float('-inf'))
 logits=raw/math.sqrt(128)+lc[None,:,:];maximum=tl.max(logits,2);maximum=tl.where(pg[None,:]<B,maximum,0.)
 w=tl.exp(logits-maximum[:,:,None]);den=tl.sum(w,2);den=tl.where(pg[None,:]<B,den,1.)
 score=maximum+tl.log(den);w=w/den[:,:,None]
 cost=tl.load(C+OFFSET+pg,pg<B,0.)
 tl.store(W+(h*R+ROFF+pg[None,:,None]*P+p[None,None,:])*N+qi[:,None,None],w*cost[None,:,None],(qi[:,None,None]<N)&(pg[None,:,None]<B))
 tl.store(S+(h*N+qi[:,None])*TOTAL+OFFSET+pg[None,:],score,(qi[:,None]<N)&(pg[None,:]<B))
class CompactWeights:
 def score(self,q,state):
  h,n,_=q.shape;b=state['costs'].numel();r=state['value'].shape[1];qq=project(q,state['prepared'][0]);score=torch.empty((h,n,b),device=q.device,dtype=torch.float32);w=torch.empty((h,r,n),device=q.device,dtype=torch.float16)
  off=roff=0
  for _,lc,_ in state['prepared'][1]:
   _,np,p=lc.shape;gp=64//p
   _part[(tr.cdiv(n,64),tr.cdiv(np,gp),h)](qq,state['padded_key'],state['padded_lc'],score,w,state['costs'],n,np,b,off,p,64,gp,r,roff,num_warps=4,num_stages=1)
   off+=np;roff+=np*p
  return score,(w,None)
