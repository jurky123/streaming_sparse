import torch,triton as tr
from .head_route import HeadRoute
from ..opt8.dual_means import means
from ..opt5.weight_route import _weighted_candidates
from ..opt4.kernels import _gram_stats
from .moment_route import MomentRoute
class TunedHeadRoute(HeadRoute):
 def stats(self,q,sketch,score,mask,rem,add,raw,state):
  weights,energy=raw;h,n,b=score.shape;r=weights.shape[1]
  stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
  cv=torch.empty((h*n,16,32),device=q.device,dtype=torch.float32)
  mean=means(weights,mask,state,bm=32,bk=64,warps=4,stages=3)
  _weighted_candidates[(h*n,)](weights,state['value'],state['starts'],state['counts'],sketch.costs,rem,add,cv,n,r,num_warps=4,enable_fp_fusion=True)
  _gram_stats[(h*n,)](mean,energy,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),num_warps=1,enable_fp_fusion=False)
  return stats,cv

class TunedMomentRoute(MomentRoute):
 stats_warps=1
