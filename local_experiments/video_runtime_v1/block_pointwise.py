"""Fuse block modulation/residual traffic while retaining each dtype rounding."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _pointwise(X,Y,S,T,O,N:tl.constexpr,D:tl.constexpr,F:tl.constexpr,SS:tl.constexpr,TS:tl.constexpr,MODULATE:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N*D;frame=i//(F*D);col=i%D
 x=tl.load(X+i,ok,0).to(tl.float32);s=tl.load(S+frame*SS+col,ok,0).to(tl.float32)
 dtype=O.dtype.element_ty
 if MODULATE:
  scale=(1.+s).to(dtype).to(tl.float32)
  product=(x*scale).to(dtype).to(tl.float32)
  shift=tl.load(T+frame*TS+col,ok,0).to(tl.float32)
  value=product+shift
 else:
  y=tl.load(Y+i,ok,0).to(tl.float32)
  product=(y*s).to(dtype).to(tl.float32)
  value=x+product
 tl.store(O+i,value,ok)

def reference(x,y,s,t,modulate):
 frames=s.shape[1];shape=(1,frames,x.shape[1]//frames,x.shape[2])
 if modulate:return (x.reshape(shape)*(1+s)+t).reshape_as(x)
 return x+(y.reshape(shape)*s).reshape_as(x)

def fused(x,y,s,t,modulate):
 assert x.is_contiguous() and x.shape[0]==1 and x.dtype==s.dtype
 assert (modulate and t.dtype==x.dtype) or (not modulate and y.dtype==x.dtype)
 out=torch.empty_like(x);n,d=x.shape[1:];frames=s.shape[1]
 _pointwise[(tr.cdiv(n*d,1024),)](x,x if y is None else y,s,s if t is None else t,out,n,d,n//frames,s.stride(1),s.stride(1) if t is None else t.stride(1),modulate,1024,num_warps=4,enable_fp_fusion=False)
 return out

class PointwiseBlocks:
 def __init__(self,engine,optimized=False,observe=None,profile=False):
  self.original={};self.engine=engine;self.optimized=optimized;self.observe=observe;self.profile=profile
  for i,b in enumerate(engine.wrapper.model.blocks):
   self.original[i]=b.forward
   def forward(x,index=i,block=b,**kw):return self.forward(index,block,x,**kw)
   b.forward=forward
 def apply(self,layer,name,x,y,s,t,modulate):
  fn=fused if self.optimized else reference
  if self.profile:
   with torch.profiler.record_function('pointwise.'+name):out=fn(x,y,s,t,modulate)
  else:out=fn(x,y,s,t,modulate)
  if self.observe is not None:self.observe(layer,name,(x,y,s,t,modulate),out)
  return out
 def norm_modulate(self,l,name,norm,x,scale,shift):
  if (getattr(self,'fuse_norm',False) and x.shape==(1,4608,1536) and x.is_contiguous()
      and x.dtype==scale.dtype==shift.dtype==torch.bfloat16 and not norm.elementwise_affine):
   from .layernorm_mod import apply
   return apply(x,scale,shift,norm.eps)
  return self.apply(l,name,norm(x),None,scale,shift,True)
 def forward(self,l,b,x,e,seq_lens,grid_sizes,freqs,context,context_lens,block_mask,kv_cache=None,crossattn_cache=None,current_start=0,cache_start=None):
  e=(b.modulation.unsqueeze(1)+e).chunk(6,dim=2)
  q=self.norm_modulate(l,'mod1',b.norm1,x,e[1],e[0])
  y=b.self_attn(q,seq_lens,grid_sizes,freqs,block_mask,kv_cache,current_start,cache_start)
  merge_norm=(getattr(self,'residual_norm',False) and self.fuse_norm and
              x.shape==(1,4608,1536) and x.is_contiguous() and y.is_contiguous() and
              x.dtype==y.dtype==e[2].dtype==torch.bfloat16)
  if (merge_norm and isinstance(b.norm3,torch.nn.LayerNorm) and b.norm3.elementwise_affine
      and b.norm3.weight.dtype==b.norm3.bias.dtype==x.dtype):
   from .layernorm_mod import gated_and_normalize
   x,normalized=gated_and_normalize(x,y,e[2],b.norm3)
  else:
   x=self.apply(l,'res1',x,y,e[2],None,False)
   normalized=b.norm3(x)
  cross=b.cross_attn(normalized,context,context_lens,crossattn_cache=crossattn_cache)
  if merge_norm and cross.is_contiguous() and cross.dtype==x.dtype and not b.norm2.elementwise_affine:
   from .layernorm_mod import add_and_modulate
   x,q=add_and_modulate(x,cross,e[4],e[3],b.norm2.eps)
  else:
   x=x+cross
   q=self.norm_modulate(l,'mod2',b.norm2,x,e[4],e[3])
  y=b.ffn(q)
  return self.apply(l,'res2',x,y,e[5],None,False)
 def close(self):
  for i,fn in self.original.items():self.engine.wrapper.model.blocks[i].forward=fn
