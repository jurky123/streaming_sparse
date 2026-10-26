"""Separate H/C candidate pools across CTAs instead of expanding one CTA."""
import torch
import triton as tr
import triton.language as tl
from ..opt9.select import _pick

@tr.jit
def _boundary(X,S,O,T,PART,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,BETA:tl.constexpr):
 r=tl.program_id(0);pool=tl.program_id(1);p=tl.arange(0,P)
 x=tl.load(X+r*B+p,p<B,0.);mean=tl.sum(x,0)/B
 std=tl.sqrt(tl.sum(tl.where(p<B,(x-mean)*(x-mean),0.),0)/B)
 selected=(p<B)&((x>mean+BETA*std)|(std<=1.e-6))
 if tl.sum(selected.to(tl.int32),0)==0:
  maximum=tl.max(tl.where(p<B,x,float('-inf')),0)
  best=tl.min(tl.where((p<B)&(x==maximum),p,2147483647),0);selected=p==best
 if pool==0:
  tl.store(S+r*B+p,selected,p<B);tl.store(T+r*2,mean);tl.store(T+r*2+1,std)
 origin=tl.load(O+p,p<B,-1);distance=tl.abs(x-mean-BETA*std)
 remain=(p<B)&tl.where(pool<2,selected,~selected)&(origin==pool%2)
 for c in tl.static_range(3):
  idx=_pick(distance,remain,p,P,'packed32');valid=idx<2147483647
  tl.store(PART+(r*4+pool)*3+c,tl.where(valid,idx,-1));remain=remain&~(valid&(p==idx))

@tr.jit
def _explore(X,S,T,PART,REM,ADD,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,BETA:tl.constexpr):
 r=tl.program_id(0);side=tl.program_id(1);p=tl.arange(0,P);cols=tl.arange(0,8)
 selected=tl.load(S+r*B+p,p<B,False);remain=(p<B)&tl.where(side==0,selected,~selected)
 result=tl.full((8,),-1,tl.int32)
 for c in tl.static_range(6):
  idx=tl.load(PART+(r*4+side*2+c//3)*3+c%3)
  result=tl.where(cols==c,idx,result);remain=remain&~((idx>=0)&(p==idx))
 q=r%N;h=r//N;tile_n=tl.minimum(128,N-q//128*128);canonical=h*tile_n+q%128
 raw=p.to(tl.uint64)*1103515245+canonical.to(tl.uint64)*12345+16
 folded=(raw&2147483647)+(raw>>31);hashes=tl.where(folded>=2147483647,folded-2147483647,folded).to(tl.float32)
 for c in tl.static_range(6,8):
  idx=_pick(hashes,remain,p,P,'packed32');valid=idx<2147483647
  result=tl.where(cols==c,tl.where(valid,idx,-1),result);remain=remain&~(valid&(p==idx))
 if tl.min(result,0)<0:
  x=tl.load(X+r*B+p,p<B,0.);mean=tl.load(T+r*2);std=tl.load(T+r*2+1);distance=tl.abs(x-mean-BETA*std)
  for c in tl.static_range(8):
   missing=tl.sum(tl.where(cols==c,result,0),0)<0
   idx=_pick(distance,remain,p,P,'packed32');valid=missing&(idx<2147483647)
   result=tl.where((cols==c)&valid,idx,result);remain=remain&~(valid&(p==idx))
 if side==0:tl.store(REM+r*8+cols,result)
 else:tl.store(ADD+r*8+cols,result)

def select(score,origins,beta,warps=4):
 h,n,b=score.shape;mask=torch.empty_like(score,dtype=torch.bool);tmp=torch.empty((h*n,2),device=score.device);part=torch.empty((h*n,4,3),device=score.device,dtype=torch.int32);rem=torch.empty((h*n,8),device=score.device,dtype=torch.int64);add=torch.empty_like(rem)
 _boundary[(h*n,4)](score,mask,origins,tmp,part,n,b,tr.next_power_of_2(b),beta,num_warps=warps,enable_fp_fusion=False)
 _explore[(h*n,2)](score,mask,tmp,part,rem,add,n,b,tr.next_power_of_2(b),beta,num_warps=warps,enable_fp_fusion=False)
 return mask,rem,add
