"""Reuse each loaded prototype weight tile for full and selected set means."""
import torch,triton as tr
import triton.language as tl
@tr.jit
def _dual(W,V,MASK,PAGE,OUT,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr):
 h=tl.program_id(1);q=tl.program_id(0)*BM+tl.arange(0,BM);k=tl.arange(0,BK);d=tl.arange(0,32)
 total=tl.zeros((BM,32),tl.float32);selected_sum=tl.zeros((BM,32),tl.float32)
 for off in range(tr.cdiv(R,BK)):
  r=off*BK+k;page=tl.load(PAGE+r,r<R,0)
  selected=tl.load(MASK+(h*N+q[:,None])*B+page[None,:],(q[:,None]<N)&(r[None,:]<R),False)
  a=tl.load(W+(h*R+r[None,:])*N+q[:,None],(q[:,None]<N)&(r[None,:]<R),0.)
  v=tl.load(V+(h*R+r[:,None])*32+d[None,:],r[:,None]<R,0.).to(tl.float16)
  total=tl.dot(a,v,total)
  selected_sum=tl.dot(tl.where(selected,a,0.).to(tl.float16),v,selected_sum)
 tl.store(OUT+(h*2*N+q[:,None])*32+d[None,:],total,q[:,None]<N)
 tl.store(OUT+(h*2*N+N+q[:,None])*32+d[None,:],selected_sum,q[:,None]<N)

def means(weights,mask,state,bm=64,bk=64,warps=4,stages=2):
 h,n,b=mask.shape;r=weights.shape[1];out=torch.empty((h,2*n,32),device=weights.device,dtype=torch.float32)
 _dual[(tr.cdiv(n,bm),h)](weights,state['value'],mask,state['page_index'],out,n,b,r,bm,bk,num_warps=warps,num_stages=stages)
 return out
