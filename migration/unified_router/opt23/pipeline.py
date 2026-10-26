"""Explicit per-chunk geometry; identical algorithms to Opt22 at SEG=4680."""
import math
from types import SimpleNamespace
import torch
import triton as tr
from ..opt16.grouping import order,gather
from ..opt16.pipeline import PackedKV
from ..opt16.route import prepare
from ..opt19.prototype import segment
from ..opt19.cache import _write
from ..opt19.tile import select
from .route import scores
from .attention import Quantized,attention

class Cache:
 def __init__(self,model,raw,seg):
  self.model=model;self.seg=seg;self.nh=raw['kh'].shape[1];self.n=raw['kc'].shape[1]
  assert self.n==seg and self.nh%seg==0 and self.nh>0
  self.par=model.pipeline.parameters;h,_,d=raw['kh'].shape
  self.k=torch.empty((h,self.nh+seg,d),device=raw['kh'].device,dtype=raw['kh'].dtype);self.v=torch.empty_like(self.k)
  self.refresh_history(raw['kh'],raw['vh'])
 def refresh_history(self,k,v):
  parts=[];p=self.par
  for start in range(0,self.nh,self.seg):
   age=(self.nh-start)//self.seg;np=8 if age==1 else 4 if age==2 else 2
   ix=order(k[:,start:start+self.seg],self.model.tree('k'));kk=gather(k[:,start:start+self.seg],ix);vv=gather(v[:,start:start+self.seg],ix)
   self.k[:,start:start+self.seg].copy_(kk);self.v[:,start:start+self.seg].copy_(vv)
   parts.append(segment(kk,p[f'history_{np}_weight'],p[f'history_{np}_bias'],p['key_basis'],half=True))
  self.history=SimpleNamespace(parts=tuple(parts))
 def current(self,k,v):
  ix=order(k,self.model.tree('k'));p=self.par
  _write[(tr.cdiv(self.n*128,256),k.shape[0])](k,v,ix,self.k,self.v,self.n,self.nh+self.n,self.nh,k.stride(0),k.stride(1),v.stride(0),v.stride(1),num_warps=4)
  pp=segment(self.k[:,self.nh:],p['current_weight'],p['current_bias'],p['key_basis'],half=True)
  return PackedKV(self.k,self.v,None,prepare((p['key_basis'][...,:64],list(self.history.parts)+[pp])),self.nh)

class Pipeline:
 def __init__(self,model,raw,seg):
  self.model=model;self.seg=seg;self.cache=Cache(model,raw,seg);self.quant=Quantized(self.cache)
 def refresh_history(self,k,v):self.cache.refresh_history(k,v);self.quant.history()
 def run(self,q,kc,vc,return_aux=False):
  kv=self.cache.current(kc,vc);self.quant.current();ix=order(q,self.model.tree('q'));qq=gather(q,ix)
  fine,candidates=scores(qq,kv.state,.35,8,self.seg);ids=select(fine,candidates,kv.state['key'].shape[1],32,.2)
  out=attention(qq,kv,ids,self.quant,self.seg,ix)
  return (out,dict(kv=kv,q_order=ix,q_packed=qq,selection=ids,fine=fine,candidates=candidates)) if return_aux else out
