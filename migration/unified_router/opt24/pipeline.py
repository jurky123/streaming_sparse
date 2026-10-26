"""Production-compatible32 routing plus persistent page-packed Sparge storage."""
import torch,triton as tr
import triton.language as tl
from .sparge import module,quant,metadata
from ..opt21.pipeline import Pipeline as Parent
from ..opt16.grouping import order,gather,restore
from ..opt19.route_fast import scores
from ..opt19.tile import select
@tr.jit
def _sync(K,V,KO,VO,N:tl.constexpr,B:tl.constexpr,START:tl.constexpr,SEG:tl.constexpr,PAGES:tl.constexpr):
 p=START+tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,128)
 i=p//PAGES*SEG+p%PAGES*64+t;valid=(i<N)&(p%PAGES*64+t<SEG);src=(h*N+i[:,None])*128+d[None,:];dst=((h*B+p)*64+t[:,None])*128+d[None,:]
 tl.store(KO+dst,tl.load(K+src,valid[:,None],0));tl.store(VO+dst,tl.load(V+src,valid[:,None],0).to(tl.float16))
class Packed:
 def __init__(self,pipe):
  self.pipe=pipe;self.module=module();self.seg=4680;self.pages=74;self.b=pipe.quant.b;h=pipe.cache.k.shape[0]
  self.k=torch.empty((h,self.b*64,128),device=pipe.cache.k.device,dtype=torch.int8);self.v=torch.empty_like(self.k,dtype=torch.float16);self.history()
 def sync(self,start,end):
  c=self.pipe.cache
  _sync[(end-start,c.k.shape[0])](self.pipe.quant.k,c.v,self.k,self.v,c.k.shape[1],self.b,start,self.seg,self.pages,num_warps=4)
 def history(self):self.sync(0,self.pipe.quant.nhpages)
 def current(self):self.sync(self.pipe.quant.nhpages,self.b)
 def attention(self,q,ids,ix):
  qi,qs=quant(q,32);lut,count=metadata(ids,self.b);out=torch.empty_like(q)
  self.module.run(qi,self.k,self.v,out,lut,count,qs,self.pipe.quant.scale,32,self.pages,8)
  return restore(out,ix)
class Pipeline(Parent):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);assert self.group==32 and self.budget==.2
  self.packed=Packed(self);old=self.quant.history
  # TileGraph.refresh_history updates owned KV/prototypes in place and invokes
  # quant.history directly. Mirror that update into the persistent paged cache.
  def history():old();self.packed.history()
  self.quant.history=history
 def run(self,q,kc,vc,return_aux=False):
  kv=self.cache.current(kc,vc);self.quant.current();self.packed.current()
  ix=order(q,self.model.tree('q'));qq=gather(q,ix)
  fine,candidates=scores(qq,kv.state,self.fraction,self.reps);chosen=select(fine,candidates,kv.state['key'].shape[1],32,self.budget)
  out=self.packed.attention(qq,chosen,ix)
  return (out,dict(selection=chosen,kv=kv,q_order=ix,fine=fine,candidates=candidates)) if return_aux else out

def install():
 from ..opt20 import network
 network.Pipeline=Pipeline
 module()
