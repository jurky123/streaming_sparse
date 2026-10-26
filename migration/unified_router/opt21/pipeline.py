"""Same route and attention math; combine split merge and query restore."""
from ..opt19.pipeline import Pipeline as Parent
from ..opt16.grouping import order,gather
from ..opt19.route_fast import scores
from ..opt19.tile import select
from .attention import attention

class Pipeline(Parent):
 def run(self,q,kc,vc,return_aux=False):
  assert self.mode=='shared' and self.group==32
  kv=self.cache.current(kc,vc);self.quant.current()
  ix=order(q,self.model.tree('q'));qq=gather(q,ix)
  fine,candidates=scores(qq,kv.state,self.fraction,self.reps)
  chosen=select(fine,candidates,kv.state['key'].shape[1],32,self.budget)
  out=attention(qq,kv,chosen,self.quant,ix,splits=2,warps=2,stages=2)
  return (out,dict(selection=chosen,kv=kv,q_order=ix,fine=fine,candidates=candidates)) if return_aux else out

def install():
 from ..opt20 import network
 network.Pipeline=Pipeline
