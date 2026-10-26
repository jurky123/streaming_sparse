import torch,triton as tr
from ..opt9.network import PackedRoute
from ..opt6.factor import _factor_prepare
from .finish import _finish
class HeadMixin:
 head_fp16=True
 approx_gelu=True
 pair_precision='fp16'
 def __init__(self,**kwargs):
  super().__init__(**kwargs);self.pair_precision='fp16'
 def actions(self,q,sketch,score,mask,rem,add,beta,step,stats,cv,head):
  h,n,b=score.shape;out=torch.empty_like(mask);tmp=torch.empty((h*n,384),device=q.device,dtype=torch.float16 if self.head_fp16 else torch.float32)
  _factor_prepare[(h*n,)](stats,score,cv,sketch.costs,sketch.origins,rem,add,head.w1,head.b1,head.mean,head.std,step,tmp,n,b,sketch.history_tokens,sketch.current_tokens,beta,self.pair_precision,num_warps=1,enable_fp_fusion=False)
  _finish[(h*n,)](tmp,stats,mask,rem,add,head.w1,head.w2,head.b2,out,n,b,tr.next_power_of_2(b),self.approx_gelu,num_warps=1,enable_fp_fusion=False)
  return out
class HeadRoute(HeadMixin,PackedRoute):pass
