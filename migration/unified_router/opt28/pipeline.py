"""Experimental tile adapter; retain Q32 coarse/fine, change final shared pages.

Every timed invocation includes current K/V and query grouping/preparation.
History is prepared once, as in the Opt27 warm-module baseline.
"""
import math,torch,triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint
from ..opt25.pipeline import SharedCache
from ..opt27.prep_probe import LargeWorkspace
from ..opt16.pipeline import PackedKV
from ..opt16.grouping import order,restore
from ..opt19.prototype import segment
from ..opt16.route import prepare
from ..opt26.route import coarse,radix_top,_top,original_fine
from ..opt24.sparge import metadata
from .native import Native
from .retile_v2 import select
@tr.jit
def _q(Q,IX,BASIS,P,QI,QS,N:tl.constexpr,T:tl.constexpr,G:tl.constexpr,QH:tl.constexpr,QN:tl.constexpr,BH:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);r=g*G+tl.arange(0,G);d=tl.arange(0,128);c=tl.arange(0,64)
 ix=tl.load(IX+h*N+r,r<N,0);q=tl.load(Q+h*QH+ix[:,None]*QN+d[None,:],r[:,None]<N,0)
 basis=tl.load(BASIS+h*BH+d[:,None]*BK+c[None,:]*BD).to(tl.float16)
 pp=tl.dot(q.to(tl.float16),basis);tl.store(P+(h*N+r[:,None])*64+c[None,:],pp,r[:,None]<N)
 scale=tl.maximum(tl.max(tl.max(tl.abs(q.to(tl.float32)),1),0)/127.,1e-10)
 tl.store(QI+(h*N+r[:,None])*128+d[None,:],nearbyint(q.to(tl.float32)/scale).to(tl.int8),r[:,None]<N);tl.store(QS+h*T+g,scale)
def query(q,ix,basis,g):
 h,n,d=q.shape;t=tr.cdiv(n,g);proj=torch.empty((h,n,64),device=q.device,dtype=torch.float16);qi=torch.empty((h,n,128),device=q.device,dtype=torch.int8);qs=torch.empty((h,t),device=q.device)
 _q[(t,h)](q,ix,basis,proj,qi,qs,n,t,g,q.stride(0),q.stride(1),basis.stride(0),basis.stride(1),basis.stride(2),num_warps=4,num_stages=1)
 return proj,qi,qs
@tr.jit
def _kv(K,V,IX,KO,VO,KS,PK,PV,N:tl.constexpr,NH:tl.constexpr,NT:tl.constexpr,B:tl.constexpr,HP:tl.constexpr,TILE:tl.constexpr,KH:tl.constexpr,KN:tl.constexpr,VH:tl.constexpr,VN:tl.constexpr):
 page=tl.program_id(0);h=tl.program_id(1);r=page*TILE+tl.arange(0,TILE);d=tl.arange(0,128)
 ix=tl.load(IX+h*N+r,r<N,0);k=tl.load(K+h*KH+ix[:,None]*KN+d[None,:],r[:,None]<N,0);v=tl.load(V+h*VH+ix[:,None]*VN+d[None,:],r[:,None]<N,0)
 old=(h*NT+NH+r[:,None])*128+d[None,:];packed=((h*B+HP+page)*TILE+tl.arange(0,TILE)[:,None])*128+d[None,:]
 tl.store(KO+old,k,r[:,None]<N);tl.store(VO+old,v,r[:,None]<N)
 scale=tl.maximum(tl.max(tl.max(tl.abs(k.to(tl.float32)),1),0)/127.,1e-10);ki=nearbyint(k.to(tl.float32)/scale).to(tl.int8)
 tl.store(KS+h*B+HP+page,scale);tl.store(PK+packed,ki);tl.store(PV+packed,v.to(tl.float16))
class Pipeline:
 def __init__(self,model,raw,aux,g,k,w,select_fn=select):
  self.select=select_fn;self.model=model;self.seg=raw['kc'].shape[1];self.cache=SharedCache(model,raw,LargeWorkspace(raw));self.lib=Native(aux['kv'],self.seg,g,k,w)
 def current(self,k,v):
  cache=self.cache;p=cache.par;lib=self.lib;ix=order(k,self.model.tree('k'));h,n,d=k.shape
  _kv[(lib.pages,h)](k,v,ix,cache.k,cache.v,lib.ks,lib.ki,lib.v,n,cache.nh,cache.nh+n,lib.b,cache.nh//n*lib.pages,lib.k,k.stride(0),k.stride(1),v.stride(0),v.stride(1),num_warps=4,num_stages=1)
  pp=segment(cache.k[:,cache.nh:],p['current_weight'],p['current_bias'],p['key_basis'],half=True)
  return PackedKV(cache.k,cache.v,None,prepare((p['key_basis'][...,:64],list(cache.history.parts)+[pp])),cache.nh)
 def scores(self,q,state):
  h,n,_=q.shape;g=tr.cdiv(n,32);b=state['key'].shape[1];c=math.ceil(b*.35)
  score=coarse(q,state,8,128 if b>1024 else 64,4)
  if b>512:candidates=radix_top(score,c,8,4)
  else:
   candidates=torch.empty((h,g,c),device=q.device,dtype=torch.int32);_top[(h*g,)](score,candidates,h*g,b,c,tr.next_power_of_2(b),num_warps=4)
  fine=torch.empty((h,n,c),device=q.device)
  original_fine[(g,tr.cdiv(c,8),h)](q,state['key'],state['logcount'],candidates,fine,n,b,c,g,tr.cdiv(self.seg,64),num_warps=4,num_stages=1)
  return fine,candidates
 def attention(self,qi,qs,ids,ix,dtype):
  lib=self.lib;lut,count=metadata(ids,lib.b);out=torch.empty(qi.shape,device=qi.device,dtype=dtype)
  lib.module.run(qi,lib.ki,lib.v,out,lut,count,qs,lib.ks,lib.g,lib.k,lib.w,lib.pages,lib.seg%lib.k or lib.k)
  return restore(out,ix)
 def run(self,q,kc,vc,return_aux=False):
  kv=self.current(kc,vc);ix=order(q,self.model.tree('q'));qq,qi,qs=query(q,ix,kv.state['basis'],self.lib.g);fine,candidates=self.scores(qq,kv.state)
  ids=self.select(fine,candidates,kv.state['key'].shape[1],self.seg,self.lib.g,self.lib.k);out=self.attention(qi,qs,ids,ix,q.dtype)
  return (out,dict(kv=kv,q_order=ix,projected=qq,qi=qi,qs=qs,fine=fine,candidates=candidates,selection=ids)) if return_aux else out
