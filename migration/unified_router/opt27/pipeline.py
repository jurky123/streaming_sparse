"""Optional exact preparation fusions, with unchanged Opt26 route and Sparge32."""
import torch
from ..opt25 import pipeline as parent
from ..opt26.pipeline import Pipeline as Original
from ..opt16.grouping import order,gather,restore
from ..opt24.sparge import quant,metadata
from ..opt8.projection import project
from .prep import query,current
from .route import route
class Pipeline(Original):
 mode='both'
 def run(self,q,kc,vc,return_aux=False):
  if not self.sparge:return super().run(q,kc,vc,return_aux)
  if self.mode in ['both','kv']:kv=current(self,kc,vc)
  else:kv=self.cache.current(kc,vc);self.quant.current();self.sync(self.quant.nhpages,self.quant.b)
  ix=order(q,self.model.tree('q'))
  if self.mode in ['both','q']:qq,qi,qs=query(q,ix,kv.state['basis'])
  else:
   ordered=gather(q,ix);qq=project(ordered,kv.state['basis']);qi,qs=quant(ordered,32)
  ids=route(qq,kv.state,self.seg);lut,count=metadata(ids,self.quant.b);out=torch.empty((q.shape[0],q.shape[1],128),device=q.device,dtype=q.dtype)
  self.module.run(qi,self.pk,self.pv,out,lut,count,qs,self.quant.scale,32,self.pages,self.seg%64 or 64)
  out=restore(out,ix)
  return (out,dict(selection=ids,kv=kv,q_order=ix)) if return_aux else out

def install(enabled=True):parent.Pipeline=Pipeline if enabled else Original
