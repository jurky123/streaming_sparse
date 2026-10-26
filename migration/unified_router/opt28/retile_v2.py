"""Materialize only small merged page scores to use validated 1D top-k layouts.

Identical probabilities and page selections to retile.py; no algorithm change.
"""
import math,torch,triton as tr
import triton.language as tl
from .retile import _importance
from ..opt19.tile import select as original
from ..opt26.radix import top as radix_top
from ..opt17.route import _top
@tr.jit
def _merge(I,S,B64:tl.constexpr,B:tl.constexpr,T0:tl.constexpr,T:tl.constexpr,R:tl.constexpr,P64:tl.constexpr,PK:tl.constexpr,KTILE:tl.constexpr,BB:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);r=t*R+tl.arange(0,R);p=tl.arange(0,BB)
 if KTILE==64:old=p
 else:old=p//PK*P64+(p%PK)*2
 x=tl.load(I+(h*T0+r[:,None])*B64+old[None,:],(r[:,None]<T0)&(p[None,:]<B),0.)
 if KTILE==128:
  x+=tl.load(I+(h*T0+r[:,None])*B64+old[None,:]+1,(r[:,None]<T0)&(p[None,:]<B)&((p[None,:]%PK)*2+1<P64),0.)
 tl.store(S+(h*T+t)*B+p,tl.sum(x,0),p<B)
def select(fine,candidates,b64,seg,g,k):
 if k==64 and g in (16,32):return original(fine,candidates,b64,g,.2)
 h,n,c=fine.shape;p64=tr.cdiv(seg,64);pk=tr.cdiv(seg,k);b=b64//p64*pk;base=16 if g==16 else 32;t0=tr.cdiv(n,base);t=tr.cdiv(n,g)
 imp=torch.zeros((h,t0,b64),device=fine.device)
 _importance[(t0,h)](fine,candidates,imp,n,b64,c,candidates.shape[1],t0,base,tr.next_power_of_2(c),num_warps=4)
 score=torch.empty((h,t,b),device=fine.device)
 _merge[(t,h)](imp,score,b64,b,t0,t,g//base,p64,pk,k,tr.next_power_of_2(b),num_warps=4)
 keep=math.ceil(b*.2)
 if b>512:return radix_top(score,keep,8,4)
 out=torch.empty((h,t,keep),device=fine.device,dtype=torch.int32)
 _top[(h*t,)](score,out,h*t,b,keep,tr.next_power_of_2(b),num_warps=4)
 return out
