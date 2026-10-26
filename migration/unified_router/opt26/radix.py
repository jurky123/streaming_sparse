"""Exact float-order radix threshold; preserve original index-order tie breaking."""
import triton as tr
import triton.language as tl
import torch
from ..opt17.route import _chosen
@tr.jit
def _radix(score,index,VALID:tl.constexpr,KEEP:tl.constexpr,BITS:tl.constexpr):
 bits=score.to(tl.uint32,bitcast=True);ordered=tl.where((bits&0x80000000)!=0,~bits,bits^0x80000000)
 active=index<VALID;k=tl.full((),KEEP,tl.int32);bins=tl.arange(0,1<<BITS)
 for step in tl.static_range(tr.cdiv(32,BITS)):
  shift=tl.maximum(32-(step+1)*BITS,0);digit=((ordered>>shift)&((1<<BITS)-1)).to(tl.int32)
  hist=tl.histogram(tl.where(active,digit,0),1<<BITS)
  hist=hist-tl.where(bins==0,tl.sum((~active).to(tl.int32),0),0)
  cumulative=tl.cumsum(hist,0);total=tl.sum(hist,0);above=total-cumulative
  picked=tl.max(tl.where((above<k)&(above+hist>=k),bins,0),0)
  k=k-tl.sum(tl.where(bins==picked,above,0),0);active=active&(digit==picked)
 cutoff=tl.max(tl.where(active,ordered,0),0)
 above=(ordered>cutoff)&(index<VALID);equal=(ordered==cutoff)&(index<VALID)
 need=KEEP-tl.sum(above.to(tl.int32),0)
 return above|(equal&(tl.cumsum(equal.to(tl.int32),0)<=need))
@tr.jit
def _top(S,I,ROWS:tl.constexpr,B:tl.constexpr,C:tl.constexpr,BB:tl.constexpr,BITS:tl.constexpr):
 row=tl.program_id(0);p=tl.arange(0,BB);score=tl.load(S+row*B+p,p<B,float('-inf'))
 chosen=_radix(score,p,B,C,BITS);rank=tl.cumsum(chosen.to(tl.int32),0)-1;tl.store(I+row*C+rank,p,chosen)
def top(score,c,bits=8,warps=4):
 if bits!=8:raise ValueError('Only the validated 8-bit radix path is supported')
 h,g,b=score.shape;out=torch.empty((h,g,c),device=score.device,dtype=torch.int32)
 _top[(h*g,)](score,out,h*g,b,c,tr.next_power_of_2(b),bits,num_warps=warps)
 return out
