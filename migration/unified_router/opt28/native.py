from pathlib import Path
import torch,triton as tr
import triton.language as tl
from ..opt24.sparge import quant,metadata
from ..opt16.grouping import restore
from .setup import ROOT,VENDOR
_module=None
def module():
 global _module
 if _module is None:
  from torch.utils.cpp_extension import load
  build=ROOT/'build_v2';build.mkdir(exist_ok=True)
  _module=load(name='opt28_tiles_v2',sources=[str(Path(__file__).with_name('binding.cu'))],extra_include_paths=[str(VENDOR/'vendor/sparge/csrc/qattn')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','-std=c++17','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--expt-relaxed-constexpr','--threads=2'],build_directory=str(build),verbose=True)
 return _module
@tr.jit
def _pack(K,V,PK,PV,N:tl.constexpr,B:tl.constexpr,SEG:tl.constexpr,PAGES:tl.constexpr,TILE:tl.constexpr):
 p=tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,TILE);d=tl.arange(0,128);i=p//PAGES*SEG+p%PAGES*TILE+t;valid=(i<N)&(p%PAGES*TILE+t<SEG)
 src=(h*N+i[:,None])*128+d[None,:];dst=((h*B+p)*TILE+t[:,None])*128+d[None,:]
 tl.store(PK+dst,tl.load(K+src,valid[:,None],0));tl.store(PV+dst,tl.load(V+src,valid[:,None],0).to(tl.float16))
class Native:
 def __init__(self,kv,seg,g,k,w):
  self.module=module();self.g,self.k,self.w,self.seg=g,k,w,seg;h,n,d=kv.k.shape;self.pages=tr.cdiv(seg,k);self.b=(n//seg)*self.pages
  pk=torch.empty((h,self.b*k,128),device=kv.k.device,dtype=kv.k.dtype);self.v=torch.empty_like(pk,dtype=torch.float16)
  _pack[(self.b,h)](kv.k,kv.v,pk,self.v,n,self.b,seg,self.pages,k,num_warps=4)
  self.ki,self.ks=quant(pk,k)
 def prepared(self,q,ids,ix):
  qi,qs=quant(q,self.g);lut,count=metadata(ids,self.b);out=torch.empty_like(q)
  self.module.run(qi,self.ki,self.v,out,lut,count,qs,self.ks,self.g,self.k,self.w,self.pages,self.seg%self.k or self.k)
  return restore(out,ix)
