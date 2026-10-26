"""Sparge SM80 CUDA reuse, tail-correct and current-stream graph compatible."""
from pathlib import Path
import torch,triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint
from .setup import ROOT
from .library import pack
from ..opt16.grouping import restore
@tr.jit
def _quant(X,Q,S,N:tl.constexpr,G:tl.constexpr,T:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);i=t*G+tl.arange(0,G);d=tl.arange(0,128)
 x=tl.load(X+(h*N+i[:,None])*128+d[None,:],i[:,None]<N,0).to(tl.float32)
 s=tl.maximum(tl.max(tl.max(tl.abs(x),1),0)/127.,1e-10)
 tl.store(Q+(h*N+i[:,None])*128+d[None,:],nearbyint(x/s).to(tl.int8),i[:,None]<N);tl.store(S+h*T+t,s)
def quant(x,g):
 h,n,d=x.shape;t=tr.cdiv(n,g);q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((h,t),device=x.device,dtype=torch.float32)
 _quant[(t,h)](x,q,s,n,g,t,num_warps=4);return q,s
@tr.jit
def _lut(I,L,C,T:tl.constexpr,K:tl.constexpr,B:tl.constexpr,KK:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);j=tl.arange(0,KK)
 page=tl.load(I+(h*T+t)*K+j,j<K,0);prev=tl.load(I+(h*T+t)*K+j-1,(j>0)&(j<K),0)
 tl.store(L+(h*T+t)*B+j,page-prev,j<K);tl.store(C+h*T+t,K)
def metadata(ids,b):
 h,t,k=ids.shape;lut=torch.empty((h,t,b),device=ids.device,dtype=torch.int32);counts=torch.empty((h,t),device=ids.device,dtype=torch.int32)
 _lut[(t,h)](ids,lut,counts,t,k,b,tr.next_power_of_2(k),num_warps=4);return lut,counts
_module=None
def module():
 global _module
 if _module is None:
  from torch.utils.cpp_extension import load
  directory=Path(__file__).parent;build=ROOT/'sparge_build';build.mkdir(exist_ok=True)
  _module=load(name='opt24_sparge_sm80',sources=[str(directory/'sparge_binding.cu')],extra_include_paths=[str(ROOT/'vendor/sparge/csrc/qattn')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','-std=c++17','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--expt-relaxed-constexpr','--threads=2'],build_directory=str(build),verbose=True)
 return _module
class Sparge:
 def __init__(self,kv,seg,group):
  self.module=module();self.seg=seg;self.group=group;self.pk,self.pv=pack(kv,seg);self.ki,self.ks=quant(self.pk,64);self.v=self.pv.half()
 def prepared(self,q,ids,ix):
  qi,qs=quant(q,self.group);lut,count=metadata(ids,self.pk.shape[1]//64);out=torch.empty_like(q)
  self.module.run(qi,self.ki,self.v,out,lut,count,qs,self.ks,self.group,tr.cdiv(self.seg,64),self.seg%64 or 64)
  return restore(out,ix)
 def run(self,q,kv,ids,ix):
  k,v=pack(kv,self.seg);ki,ks=quant(k,64);vf=v.half();qi,qs=quant(q,self.group);lut,count=metadata(ids,k.shape[1]//64);out=torch.empty_like(q)
  self.module.run(qi,ki,vf,out,lut,count,qs,ks,self.group,tr.cdiv(self.seg,64),self.seg%64 or 64);return restore(out,ix)
