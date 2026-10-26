"""Drop low-vote pages from a group, retaining each query's own mask.

Threshold scales with the valid rows in the final partial tile. If filtering
would orphan a query, restore its original pages. No PV skipping or FP16 PV.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _vote(M,O,Q:tl.constexpr,B:tl.constexpr,P:tl.constexpr,G:tl.constexpr,VOTES:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);q=g*G+tl.arange(0,G);p=tl.arange(0,P)
 original=tl.load(M+(h*Q+q[:,None])*B+p[None,:],(q[:,None]<Q)&(p[None,:]<B),False)
 votes=tl.sum(original.to(tl.int32),0)
 threshold=tl.cdiv(VOTES*tl.minimum(G,Q-g*G),G)
 kept=original&(votes[None,:]>=threshold)
 orphan=(tl.sum(kept.to(tl.int32),1)==0)&(q<Q)
 kept=kept|(original&orphan[:,None])
 tl.store(O+(h*Q+q[:,None])*B+p[None,:],kept,(q[:,None]<Q)&(p[None,:]<B))

def vote(mask,minimum=2,group=32):
 if minimum==1:return mask
 h,n,b=mask.shape;out=torch.empty_like(mask)
 _vote[(tr.cdiv(n,group),h)](mask,out,n,b,tr.next_power_of_2(b),group,minimum,num_warps=4)
 return out

class VotingRoute:
 def __init__(self,route,minimum):self.route=route;self.minimum=minimum
 def __call__(self,*args,**kwargs):return vote(self.route(*args,**kwargs),self.minimum)
