import torch,triton as tr
from ..opt9.network import PackedRoute
from ..opt8.projection import project
from .content_score import _score_content
from .content_stats import _content_stats
class ContentRoute(PackedRoute):
 quant=False
 def begin(self,prepared,sketch):
  state=super().begin(prepared,sketch)
  if self.quant:state['content_scale']=state['padded_value'].float().abs().amax((2,3)).div(127).clamp_min(1.e-10)
  return state
 def score(self,q,state):
  h,n,_=q.shape;b=state['costs'].numel();qq=project(q,state['prepared'][0]);score=torch.empty((h,n,b),device=q.device,dtype=torch.float32);energy=torch.empty_like(score)
  content=torch.empty((h,n,b,32),device=q.device,dtype=torch.int8 if self.quant else torch.float16)
  _score_content[(tr.cdiv(n,self.qm),tr.cdiv(b,self.page_group),h)](qq,state['padded_key'],state['padded_lc'],state['grams'],state['padded_value'],score,energy,state.get('content_scale',score),n,b,self.qm,self.page_group,True,content,None,None,None,0,False,self.quant,num_warps=4,num_stages=1,enable_fp_fusion=True)
  return score,(content,energy)
 def stats(self,q,sketch,score,mask,rem,add,raw,state):
  content,energy=raw;h,n,b=score.shape;stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32);cv=torch.empty((h*n,16,32),device=q.device,dtype=torch.float32)
  _content_stats[(h*n,)](content,energy,score,mask,sketch.costs,sketch.origins,stats,rem,add,cv,state.get('content_scale',score),n,b,tr.next_power_of_2(b),self.quant,num_warps=4,enable_fp_fusion=False)
  return stats,cv
