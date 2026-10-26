import torch,triton as tr
from .tuned_route import TunedHeadRoute,TunedMomentRoute
from ..opt8.dual_means import means
from ..opt4.kernels import _gram_stats
from .moment_means import _dual
from .moment_stats import _gram_stats as _moment_stats
from .fused_prepare import _factor_prepare
from .finish import _finish
class FusedActions:
 prepare_warps=4
 def actions(self,q,sketch,score,mask,rem,add,beta,step,stats,cv,head):
  weights,value,starts,counts=cv;h,n,b=score.shape;out=torch.empty_like(mask);tmp=torch.empty((h*n,384),device=q.device,dtype=torch.float16)
  _factor_prepare[(h*n,)](stats,score,value,weights,starts,counts,sketch.costs,sketch.origins,rem,add,head.w1,head.b1,head.mean,head.std,step,tmp,n,b,sketch.history_tokens,sketch.current_tokens,beta,'fp16',weights.shape[1],num_warps=self.prepare_warps,enable_fp_fusion=False)
  _finish[(h*n,)](tmp,stats,mask,rem,add,head.w1,head.w2,head.b2,out,n,b,tr.next_power_of_2(b),True,num_warps=1,enable_fp_fusion=False)
  return out
class FusedHeadRoute(FusedActions,TunedHeadRoute):
 def stats(self,q,sketch,score,mask,rem,add,raw,state):
  weights,energy=raw;h,n,b=score.shape;stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
  mean=means(weights,mask,state,bm=32,bk=64,warps=4,stages=3)
  _gram_stats[(h*n,)](mean,energy,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),num_warps=1,enable_fp_fusion=False)
  return stats,(weights,state['value'],state['starts'],state['counts'])
class FusedMomentRoute(FusedActions,TunedMomentRoute):
 def stats(self,q,sketch,score,mask,rem,add,raw,state):
  weights,_=raw;h,n,b=score.shape;r=weights.shape[1];mean=torch.empty((h,2*n,64),device=q.device,dtype=torch.float32);stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
  _dual[(tr.cdiv(n,self.bm),h)](weights,state['moments'],mask,state['page_index'],mean,n,b,r,self.bm,self.bk,num_warps=4,num_stages=self.stages)
  _moment_stats[(h*n,)](mean,state['moment_scale'],score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),num_warps=1,enable_fp_fusion=False)
  return stats,(weights,state['value'],state['starts'],state['counts'])
