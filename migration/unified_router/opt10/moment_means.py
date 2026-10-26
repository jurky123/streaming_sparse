"""Reuse each loaded prototype weight tile for full and selected set means."""
import torch,triton as tr
import triton.language as tl
@tr.jit
def _dual(W,V,MASK,PAGE,OUT,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr):
 h=tl.program_id(1);q=tl.program_id(0)*BM+tl.arange(0,BM);k=tl.arange(0,BK);d=tl.arange(0,64)
 total=tl.zeros((BM,64),tl.float32);selected_sum=tl.zeros((BM,64),tl.float32)
 for off in range(tr.cdiv(R,BK)):
  r=off*BK+k;page=tl.load(PAGE+r,r<R,0)
  selected=tl.load(MASK+(h*N+q[:,None])*B+page[None,:],(q[:,None]<N)&(r[None,:]<R),False)
  a=tl.load(W+(h*R+r[None,:])*N+q[:,None],(q[:,None]<N)&(r[None,:]<R),0.)
  v=tl.load(V+(h*R+r[:,None])*64+d[None,:],r[:,None]<R,0.).to(tl.float16)
  total=tl.dot(a,v,total)
  selected_sum=tl.dot(tl.where(selected,a,0.).to(tl.float16),v,selected_sum)
 tl.store(OUT+(h*2*N+q[:,None])*64+d[None,:],total,q[:,None]<N)
 tl.store(OUT+(h*2*N+N+q[:,None])*64+d[None,:],selected_sum,q[:,None]<N)

