"""Optional Opt25 runtime integration: change only routing kernels."""
import torch
from ..opt25 import pipeline as parent
from ..opt25.pipeline import Pipeline as Original
from ..opt16.grouping import order,gather,restore
from ..opt24.sparge import quant,metadata
from .route import optimized
class Pipeline(Original):
 def run(self,q,kc,vc,return_aux=False):
  if not self.sparge:return super().run(q,kc,vc,return_aux)
  kv=self.cache.current(kc,vc);self.quant.current();self.sync(self.quant.nhpages,self.quant.b)
  ix=order(q,self.model.tree('q'));qq=gather(q,ix);ids=optimized(qq,kv.state,self.seg)
  qi,qs=quant(qq,32);lut,count=metadata(ids,self.quant.b);out=torch.empty_like(qq)
  self.module.run(qi,self.pk,self.pv,out,lut,count,qs,self.quant.scale,32,self.pages,self.seg%64 or 64)
  out=restore(out,ix)
  return (out,dict(selection=ids,kv=kv,q_order=ix)) if return_aux else out

def install(enabled=True):parent.Pipeline=Pipeline if enabled else Original
