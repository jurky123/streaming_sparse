import torch,triton as tr
from ..opt9.network import PackedRoute
from ..opt8.projection import project
from ..opt4.kernels import _gram_stats
from .block_score import _score_energy
from .block_means import means
from .block_candidates import _weighted_candidates
class BlockRoute(PackedRoute):
 qt=32
 def score(self,q,state):
  h,n,_=q.shape;b=state['costs'].numel();r=state['value'].shape[1]
  qq=project(q,state['prepared'][0]);score=torch.empty((h,n,b),device=q.device,dtype=torch.float32);energy=torch.empty_like(score)
  weights=torch.empty((h,tr.cdiv(n,self.qt),r,self.qt),device=q.device,dtype=torch.float16)
  _score_energy[(tr.cdiv(n,self.qm),tr.cdiv(b,self.page_group),h)](qq,state['padded_key'],state['padded_lc'],state['grams'],state['padded_value'],score,energy,n,b,self.qm,self.page_group,True,weights,state['starts'],state['counts'],state['costs'],r,True,self.qt,num_warps=4,num_stages=1,enable_fp_fusion=True)
  return score,(weights,energy)
 def stats(self,q,sketch,score,mask,rem,add,raw,state):
  weights,energy=raw;h,n,b=score.shape;r=weights.shape[2]
  stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32);cv=torch.empty((h*n,16,32),device=q.device,dtype=torch.float32)
  mean=means(weights,mask,state,bm=32,bk=64,warps=4,stages=3)
  _weighted_candidates[(h*n,)](weights,state['value'],state['starts'],state['counts'],sketch.costs,rem,add,cv,n,r,self.qt,num_warps=4,enable_fp_fusion=True)
  _gram_stats[(h*n,)](mean,energy,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),num_warps=4,enable_fp_fusion=False)
  return stats,cv
