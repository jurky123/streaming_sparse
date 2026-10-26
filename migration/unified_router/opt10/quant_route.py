"""Approximate page energy through weighted prototype moments; no page-PV pass."""
import torch,triton as tr
from ..opt9.network import PackedRoute
from ..opt8.projection import project
from .quant_candidates import _weighted_candidates
from .head_route import HeadMixin
from .quant_score import _score_energy
from .quant_means import _dual
from .moment_stats import _gram_stats

def prepare_moments(value):
 norm=value.half().float().square().sum(-1)
 aug=torch.cat((value,norm[...,None],torch.zeros_like(value[...,:31])),dim=-1)
 scale=(aug.abs().amax(1)/127.).clamp_min(1.e-10)
 quant=(aug/scale[:,None,:]).round().clamp(-127,127).to(torch.int8)
 return quant,scale
class QuantRoute(HeadMixin,PackedRoute):
 bm=32;bk=64;stages=3
 def __init__(self,**kwargs):
  super().__init__(**kwargs);self.prepare_moments=torch.compile(prepare_moments,fullgraph=True)
 def begin(self,prepared,sketch):
  state=super().begin(prepared,sketch);state['moments'],state['quant_scale']=self.prepare_moments(state['value']);state['moment_scale']=torch.ones((state['value'].shape[0],),device=state['value'].device);return state
 def score(self,q,state):
  h,n,_=q.shape;b=state['costs'].numel();r=state['value'].shape[1];qq=project(q,state['prepared'][0]);score=torch.empty((h,n,b),device=q.device,dtype=torch.float32);weights=torch.empty((h,r,n),device=q.device,dtype=torch.int8)
  _score_energy[(tr.cdiv(n,self.qm),tr.cdiv(b,self.page_group),h)](qq,state['padded_key'],state['padded_lc'],None,None,score,None,n,b,self.qm,self.page_group,False,weights,state['starts'],state['counts'],state['costs'],r,True,num_warps=4,num_stages=1,enable_fp_fusion=True)
  return score,(weights,None)
 def stats(self,q,sketch,score,mask,rem,add,raw,state):
  weights,_=raw;h,n,b=score.shape;r=weights.shape[1];mean=torch.empty((h,2*n,64),device=q.device,dtype=torch.float32);stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32);cv=torch.empty((h*n,16,32),device=q.device,dtype=torch.float32)
  _dual[(tr.cdiv(n,self.bm),h)](weights,state['moments'],state['quant_scale'],mask,state['page_index'],mean,n,b,r,self.bm,self.bk,num_warps=4,num_stages=self.stages)
  _weighted_candidates[(h*n,)](weights,state['value'],state['starts'],state['counts'],sketch.costs,rem,add,cv,n,r,num_warps=4,enable_fp_fusion=True)
  _gram_stats[(h*n,)](mean,state['moment_scale'],score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),num_warps=1,enable_fp_fusion=False)
  return stats,cv
