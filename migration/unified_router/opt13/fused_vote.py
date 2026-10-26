"""Fuse voting, coverage fallback and union compaction in one kernel."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _vote_union(M,O,IDS,COUNTS,Q:tl.constexpr,B:tl.constexpr,P:tl.constexpr,G:tl.constexpr,VOTES:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);q=g*G+tl.arange(0,G);p=tl.arange(0,P)
 original=tl.load(M+(h*Q+q[:,None])*B+p[None,:],(q[:,None]<Q)&(p[None,:]<B),False)
 votes=tl.sum(original.to(tl.int32),0)
 threshold=tl.cdiv(VOTES*tl.minimum(G,Q-g*G),G)
 kept=original&(votes[None,:]>=threshold)
 orphan=(tl.sum(kept.to(tl.int32),1)==0)&(q<Q)
 kept=kept|(original&orphan[:,None])
 tl.store(O+(h*Q+q[:,None])*B+p[None,:],kept,(q[:,None]<Q)&(p[None,:]<B))
 used=tl.sum(kept.to(tl.int32),0)>0
 rank=tl.cumsum(used.to(tl.int32),0)-1
 tl.store(IDS+(h*tr.cdiv(Q,G)+g)*B+rank,p,used)
 tl.store(COUNTS+h*tr.cdiv(Q,G)+g,tl.sum(used.to(tl.int32),0))

def vote_union(mask,minimum=4,group=32):
 h,n,b=mask.shape;out=torch.empty_like(mask)
 ids=torch.empty((h,tr.cdiv(n,group),b),device=mask.device,dtype=torch.int32);counts=torch.empty(ids.shape[:2],device=mask.device,dtype=torch.int32)
 _vote_union[(tr.cdiv(n,group),h)](mask,out,ids,counts,n,b,tr.next_power_of_2(b),group,minimum,num_warps=4)
 # The route contract remains a Tensor; ownership travels with that tensor.
 out._opt13_union=(ids,counts)
 return out

class FusedVotingRoute:
 def __init__(self,route,minimum):self.route=route;self.minimum=minimum
 def __call__(self,*args,**kwargs):return vote_union(self.route(*args,**kwargs),self.minimum)
