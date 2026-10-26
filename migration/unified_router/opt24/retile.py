"""Fixed20% retile: sum the original per-query normalized proxy importance."""
import math,torch,triton as tr
import triton.language as tl
from ..opt19.tile import select as original
from ..opt17.route import _chosen
@tr.jit
def _importance(S,IDS,O,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,T:tl.constexpr,CC:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);qi=t*32+tl.arange(0,32);c=tl.arange(0,CC)
 x=tl.load(S+(h*N+qi[:,None])*C+c[None,:],(qi[:,None]<N)&(c[None,:]<C),float('-inf'))
 mx=tl.max(x,1);mx=tl.where(qi<N,mx,0.);e=tl.exp(x-mx[:,None]);prob=e/tl.maximum(tl.sum(e,1),1e-20)[:,None]
 imp=tl.sum(prob,0);page=tl.load(IDS+(h*T+t)*C+c,c<C,0)
 tl.store(O+(h*T+t)*B+page,imp,c<C)
@tr.jit
def _select(I,O,N:tl.constexpr,B:tl.constexpr,T32:tl.constexpr,T:tl.constexpr,R:tl.constexpr,K:tl.constexpr,BB:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);r=t*R+tl.arange(0,R);p=tl.arange(0,BB)
 x=tl.load(I+(h*T32+r[:,None])*B+p[None,:],(r[:,None]<T32)&(p[None,:]<B),0.)
 s=tl.where(p<B,tl.sum(x,0),float('-inf'));chosen=_chosen(s,p,B,K);rank=tl.cumsum(chosen.to(tl.int32),0)-1
 tl.store(O+(h*T+t)*K+rank,p,chosen)
def select(fine,candidates,b,group):
 if group in (16,32):return original(fine,candidates,b,group,.2)
 assert group in (64,128)
 h,n,c=fine.shape;t32=candidates.shape[1];t=tr.cdiv(n,group);k=math.ceil(b*.2)
 imp=torch.zeros((h,t32,b),device=fine.device)
 _importance[(t32,h)](fine,candidates,imp,n,b,c,t32,tr.next_power_of_2(c),num_warps=4)
 out=torch.empty((h,t,k),device=fine.device,dtype=torch.int32)
 _select[(t,h)](imp,out,n,b,t32,t,group//32,k,tr.next_power_of_2(b),num_warps=4)
 return out
