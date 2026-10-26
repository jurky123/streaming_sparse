"""Warm module with fixed native attention tiles and reduced-query routing."""
import torch
from . import kernels
from ..opt23.pipeline import Cache
from ..opt28.pipeline import Pipeline as Base,query
from ..opt28.native import Native,_pack
from ..opt24.sparge import quant
from ..opt16.grouping import order
from ..opt8.projection import project
class Pipeline(Base):
 def __init__(self,model,raw,workspace=None):
  self.model=model;self.o=model.options;self.seg=raw['kc'].shape[1];self.cache=Cache(model,raw,self.seg)if workspace is None else __import__('migration.unified_router.opt25.pipeline',fromlist=['SharedCache']).SharedCache(model,raw,workspace)
  kv=self.cache.current(raw['kc'],raw['vc']);o=self.o;self.lib=Native(kv,self.seg,o['g'],o['k'],o['warp_q'])
  if workspace is not None:
   lib=self.lib;h=raw['kc'].shape[0];maximum=h*4*lib.pages*lib.k*128
   for label in ['ki','v','ks']:
    old=getattr(lib,label);size=maximum if label!='ks'else h*4*lib.pages;new=workspace.view('native_'+label,old.shape,old.dtype,size);new.copy_(old);setattr(lib,label,new)
  self.assignments=torch.empty((raw['kc'].shape[0],self.seg,o['r']),device=raw['kc'].device,dtype=torch.float16);self.refresh_assignment=True
  if workspace is not None:self.assignments=workspace.view('query_assignments',self.assignments.shape,torch.float16,self.assignments.numel())
 def refresh_history(self,k,v):
  old=self.cache.history;self.cache.refresh_history(k,v);fresh=self.cache.history
  for a,b in zip(old.parts,fresh.parts):
   for x,y in zip(a,b):
    if x is not None:x.copy_(y)
  self.cache.history=old
  lib=self.lib;h,n,d=self.cache.k.shape;pk=torch.empty_like(lib.ki,dtype=self.cache.k.dtype)
  _pack[(lib.b,h)](self.cache.k,self.cache.v,pk,lib.v,n,lib.b,self.seg,lib.pages,lib.k,num_warps=4)
  ki,ks=quant(pk,lib.k);lib.ki.copy_(ki);lib.ks.copy_(ks)
 def run(self,q,kc,vc,return_aux=False,previous_q=None):
  o=self.o;kv=self.current(kc,vc);ix=order(q,self.model.tree('q'));qq,qi,qs=query(q,ix,kv.state['basis'],o['g'])
  a=None
  if o['mode']=='learned':
   if previous_q is not None:kernels.assign(project(previous_q,kv.state['basis']),None,self.model.query_weight,self.model.query_bias,self.assignments)
   elif self.refresh_assignment:kernels.assign(qq,ix,self.model.query_weight,self.model.query_bias,self.assignments)
   a=self.assignments
  bar,mass=kernels.aggregate(qq,ix,a,o['g'],o['r'],o['mode']);ids,d=kernels.route(qq,kv.state,bar,mass,self.seg,o['g'],o['k'],o['candidate'])
  out=self.attention(qi,qs,ids,ix,q.dtype)
  return (out,dict(d,kv=kv,q_order=ix,projected=qq,selection=ids))if return_aux else out
