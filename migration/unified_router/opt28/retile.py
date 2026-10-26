"""Common fixed-20% selection proxy for comparing hardware tile geometries.

It changes shared selection for enlarged Q/K tiles; no quality claim or training.
"""
import math,torch,triton as tr
import triton.language as tl
from ..opt19.tile import select as original
from ..opt17.route import _chosen
@tr.jit
def _importance(S,IDS,O,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,T32:tl.constexpr,T:tl.constexpr,G:tl.constexpr,CC:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);qi=t*G+tl.arange(0,G);c=tl.arange(0,CC)
 x=tl.load(S+(h*N+qi[:,None])*C+c[None,:],(qi[:,None]<N)&(c[None,:]<C),float('-inf'));mx=tl.max(x,1);mx=tl.where(qi<N,mx,0.)
 e=tl.exp(x-mx[:,None]);prob=e/tl.maximum(tl.sum(e,1),1e-20)[:,None];imp=tl.sum(prob,0)
 page=tl.load(IDS+(h*T32+t*G//32)*C+c,c<C,0);tl.store(O+(h*T+t)*B+page,imp,c<C)
@tr.jit
def _select(I,O,B64:tl.constexpr,B:tl.constexpr,T0:tl.constexpr,T:tl.constexpr,R:tl.constexpr,K:tl.constexpr,P64:tl.constexpr,PK:tl.constexpr,KTILE:tl.constexpr,BB:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);r=t*R+tl.arange(0,R);p=tl.arange(0,BB)
 if KTILE==64:old=p
 else:old=p//PK*P64+(p%PK)*2
 x=tl.load(I+(h*T0+r[:,None])*B64+old[None,:],(r[:,None]<T0)&(p[None,:]<B),0.)
 if KTILE==128:
  x+=tl.load(I+(h*T0+r[:,None])*B64+old[None,:]+1,(r[:,None]<T0)&(p[None,:]<B)&((p[None,:]%PK)*2+1<P64),0.)
 score=tl.where(p<B,tl.sum(x,0),float('-inf'))
 chosen=_chosen(score,p,B,K)
 rank=tl.cumsum(chosen.to(tl.int32),0)-1;tl.store(O+(h*T+t)*K+rank,p,chosen)
def select(fine,candidates,b64,seg,g,k):
 if k==64 and g in (16,32):return original(fine,candidates,b64,g,.2)
 h,n,c=fine.shape;p64=tr.cdiv(seg,64);pk=tr.cdiv(seg,k);b=b64//p64*pk;base=16 if g==16 else 32;t0=tr.cdiv(n,base);t=tr.cdiv(n,g)
 imp=torch.zeros((h,t0,b64),device=fine.device)
 _importance[(t0,h)](fine,candidates,imp,n,b64,c,candidates.shape[1],t0,base,tr.next_power_of_2(c),num_warps=4)
 out=torch.empty((h,t,math.ceil(b*.2)),device=fine.device,dtype=torch.int32)
 _select[(t,h)](imp,out,b64,b,t0,t,g//base,out.shape[-1],p64,pk,k,tr.next_power_of_2(b),num_warps=4)
 return out
