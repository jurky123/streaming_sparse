"""Fixed-Q32 pipeline with bounded workspaces shared across history shapes.

Graphs execute in temporal order; every chunk refreshes history and current KV
before using this storage. Views remain contiguous with their own shape strides.
"""
import torch
import triton as tr
from ..opt23.pipeline import Pipeline as Base,Cache
from ..opt23.attention import Quantized
from ..opt24.pipeline import _sync
from ..opt24.sparge import module,quant,metadata
from ..opt16.grouping import order,gather,restore
from ..opt23.route import scores
from ..opt19.tile import select

class Workspace:
 def __init__(self,raw):self.h,self.seg,self.d=raw['kc'].shape;self.raw=raw['kc'];self.buffers={}
 def view(self,name,shape,dtype,max_elements):
  if name not in self.buffers:self.buffers[name]=torch.empty(max_elements,device=self.raw.device,dtype=dtype)
  return self.buffers[name][:__import__('math').prod(shape)].view(shape)

class SharedCache(Cache):
 def __init__(self,model,raw,workspace):
  self.model=model;self.seg=raw['kc'].shape[1];self.nh=raw['kh'].shape[1];self.n=self.seg;self.par=model.pipeline.parameters
  h,_,d=raw['kc'].shape;shape=(h,self.nh+self.n,d);maximum=h*4*self.seg*d
  self.k=workspace.view('k',shape,raw['kc'].dtype,maximum);self.v=workspace.view('v',shape,raw['vc'].dtype,maximum)
  self.refresh_history(raw['kh'],raw['vh'])
class SharedQuantized(Quantized):
 def __init__(self,cache,workspace):
  self.cache=cache;self.seg=cache.seg;h,n,d=cache.k.shape;self.b=tr.cdiv(n,self.seg)*tr.cdiv(self.seg,64)
  self.k=workspace.view('ki',(h,n,d),torch.int8,h*4*self.seg*d)
  self.scale=torch.empty((h,self.b),device=cache.k.device,dtype=torch.float32)
  self.nhpages=cache.nh//self.seg*tr.cdiv(self.seg,64);self.history()

class Pipeline(Base):
 def __init__(self,model,raw,sparge,workspace):
  self.model=model;self.seg=raw['kc'].shape[1];self.cache=SharedCache(model,raw,workspace);self.quant=SharedQuantized(self.cache,workspace);self.sparge=sparge
  if sparge:
   self.module=module();h=raw['kc'].shape[0];self.pages=tr.cdiv(self.seg,64);shape=(h,self.quant.b*64,128);maximum=h*4*self.pages*64*128
   self.pk=workspace.view('pk',shape,torch.int8,maximum);self.pv=workspace.view('pv',shape,torch.float16,maximum);self.sync(0,self.quant.nhpages)
 def sync(self,start,end):
  _sync[(end-start,self.cache.k.shape[0])](self.quant.k,self.cache.v,self.pk,self.pv,self.cache.k.shape[1],self.quant.b,start,self.seg,self.pages,num_warps=4)
 def run(self,q,kc,vc,return_aux=False):
  if not self.sparge:return super().run(q,kc,vc,return_aux)
  kv=self.cache.current(kc,vc);self.quant.current();self.sync(self.quant.nhpages,self.quant.b)
  ix=order(q,self.model.tree('q'));qq=gather(q,ix);fine,candidates=scores(qq,kv.state,.35,8,self.seg)
  ids=select(fine,candidates,kv.state['key'].shape[1],32,.2)
  qi,qs=quant(qq,32);lut,count=metadata(ids,self.quant.b);out=torch.empty_like(qq)
  self.module.run(qi,self.pk,self.pv,out,lut,count,qs,self.quant.scale,32,self.pages,self.seg%64 or 64)
  out=restore(out,ix)
  return (out,dict(selection=ids,kv=kv,q_order=ix,fine=fine,candidates=candidates)) if return_aux else out

class PipelineState:
 sparge=False
 workspaces={}
 def __init__(self,model,raw):
  key=id(model)
  if key not in self.workspaces:self.workspaces[key]=Workspace(raw)
  self.pipe=Pipeline(model,raw,self.sparge,self.workspaces[key]);self.identity=None
 def refresh_history(self,kh,vh,identity):
  cache=self.pipe.cache;old=cache.history
  cache.refresh_history(kh,vh);fresh=cache.history
  for a,b in zip(old.parts,fresh.parts):
   for x,y in zip(a,b):
    if x is not None:x.copy_(y)
  cache.history=old;self.pipe.quant.history()
  if self.pipe.sparge:self.pipe.sync(0,self.pipe.quant.nhpages)
  self.identity=identity
