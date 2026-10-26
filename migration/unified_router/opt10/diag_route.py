import torch,triton as tr
from ..opt9.network import PackedRoute
from ..opt8.projection import project
from .diag_score import _score_energy
class DiagRoute(PackedRoute):
 def __init__(self,**kwargs):
  super().__init__(**kwargs);self.norms=torch.compile(lambda v:v.float().square().sum(-1),fullgraph=True)
 def begin(self,prepared,sketch):
  state=super().begin(prepared,sketch);state['norms']=self.norms(state['padded_value']);return state
 def score(self,q,state):
  h,n,_=q.shape;b=state['costs'].numel();r=state['value'].shape[1]
  qq=project(q,state['prepared'][0])
  score=torch.empty((h,n,b),device=q.device,dtype=torch.float32);energy=torch.empty_like(score)
  weights=torch.empty((h,r,n),device=q.device,dtype=torch.float16)
  _score_energy[(tr.cdiv(n,self.qm),tr.cdiv(b,self.page_group),h)](qq,state['padded_key'],state['padded_lc'],state['grams'],state['norms'],score,energy,n,b,self.qm,self.page_group,self.energy=='content',weights,state['starts'],state['counts'],state['costs'],r,True,num_warps=4,num_stages=1,enable_fp_fusion=True)
  return score,(weights,energy)
