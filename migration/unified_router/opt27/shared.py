import math
from pathlib import Path
import torch,triton as tr
import triton.language as tl
from .setup import ROOT
from ..opt17.route import _chosen
from ..opt26.radix import _radix
_module=None
def module():
 global _module
 if _module is None:
  from torch.utils.cpp_extension import load
  path=ROOT/'shared_build';path.mkdir(exist_ok=True)
  _module=load(name='opt27_shared',sources=[str(Path(__file__).with_name('shared.cu'))],build_directory=str(path),extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','-std=c++17','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__'],verbose=True)
 return _module
@tr.jit
def _select(PART,IDS,OUT,C:tl.constexpr,G:tl.constexpr,KEEP:tl.constexpr,CC:tl.constexpr,M:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);c=tl.arange(0,CC);imp=tl.load(PART+((h*G+g)*(32//M))*C+c,c<C,0)
 if M==16:imp+=tl.load(PART+((h*G+g)*2+1)*C+c,c<C,0)
 imp=tl.where(c<C,imp,float('-inf'))
 if C>512:chosen=_radix(imp,c,C,KEEP,8)&(c<C)
 else:chosen=_chosen(imp,c,C,KEEP)&(c<C)
 page=tl.load(IDS+(h*G+g)*C+c,c<C,0);rank=tl.cumsum(chosen.to(tl.int32),0)-1;tl.store(OUT+(h*G+g)*KEEP+rank,page,chosen)
def refine(qq,state,ids,seg,m=16,gp=8,warps=4):
 h,g,c=ids.shape;b=state['key'].shape[1];keep=math.ceil(b*.2);part=torch.empty((h,g,32//m,c),device=qq.device);out=torch.empty((h,g,keep),device=qq.device,dtype=torch.int32)
 module().run(qq,state['key'],state['logcount'],ids,part,tr.cdiv(seg,64),m,gp,warps)
 _select[(g,h)](part,ids,out,c,g,keep,tr.next_power_of_2(c),m,num_warps=4)
 return out
