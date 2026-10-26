"""Larger execution tiles, preserving every original32 selection and Q scale."""
import torch,triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint
from ..opt19.library import _used
from ..opt21.attention import _merge_restore
@tr.jit
def _join(U,I,M,C,T32:tl.constexpr,T:tl.constexpr,B:tl.constexpr,K:tl.constexpr,R:tl.constexpr,BB:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);r=tl.arange(0,R);p=tl.arange(0,BB)
 selected=tl.load(U+((h*T32+t*R+r[:,None])*B)+p[None,:],(t*R+r[:,None]<T32)&(p[None,:]<B),False)
 member=tl.sum(selected.to(tl.int32)*(1<<r[:,None]),0);active=(member>0)&(p<B);rank=tl.cumsum(active.to(tl.int32),0)-1
 tl.store(I+(h*T+t)*K+rank,p,active);tl.store(M+(h*T+t)*K+rank,member,active);tl.store(C+h*T+t,tl.sum(active.to(tl.int32),0))
def join(ids,b,n,group):
 h,t32,k=ids.shape;r=group//32;t=tr.cdiv(n,group);capacity=r*k
 used=torch.zeros((h,t32,b),device=ids.device,dtype=torch.bool)
 _used[(t32,h)](ids,used,t32,k,b,tr.next_power_of_2(k),num_warps=4)
 out=torch.empty((h,t,capacity),device=ids.device,dtype=torch.int32);mem=torch.empty_like(out);count=torch.empty((h,t),device=ids.device,dtype=torch.int32)
 _join[(t,h)](used,out,mem,count,t32,t,b,capacity,r,tr.next_power_of_2(b),num_warps=4)
 return out,mem,count
@tr.jit
def _att(Q,K,V,KS,IDS,MEM,COUNT,O,LSE,N:tl.constexpr,NK:tl.constexpr,B:tl.constexpr,G:tl.constexpr,T:tl.constexpr,CAP:tl.constexpr,S:tl.constexpr,SEG:tl.constexpr,INT8:tl.constexpr):
 tile=tl.program_id(0);h=tl.program_id(1);split=tl.program_id(2);r=tl.arange(0,G);qi=tile*G+r;d=tl.arange(0,128);t=tl.arange(0,64);pages=tr.cdiv(SEG,64)
 q=tl.load(Q+(h*N+qi[:,None])*128+d[None,:],qi[:,None]<N,0)
 if INT8:
  qq=q.to(tl.float32).reshape((G//32,32,128));sc=tl.maximum(tl.max(tl.max(tl.abs(qq),2),1)/127.,1e-10)
  qs=tl.broadcast_to(sc[:,None],(G//32,32)).reshape((G,));q=nearbyint(q.to(tl.float32)/qs[:,None]).to(tl.int8)
 maximum=tl.full((G,),float('-inf'),tl.float32);denom=tl.zeros((G,),tl.float32);acc=tl.zeros((G,128),tl.float32)
 count=tl.load(COUNT+h*T+tile);per=tl.cdiv(count,S)
 for j in range(split*per,tl.minimum((split+1)*per,count)):
  page=tl.load(IDS+(h*T+tile)*CAP+j);member=tl.load(MEM+(h*T+tile)*CAP+j);ki=page//pages*SEG+page%pages*64+t;valid=(ki<NK)&(page%pages*64+t<SEG)
  k=tl.load(K+(h*NK+ki[None,:])*128+d[:,None],valid[None,:],0)
  logits=tl.dot(q,k).to(tl.float32)
  if INT8:logits=logits*(qs[:,None]*tl.load(KS+h*B+page))*0.12751743082459617
  else:logits=logits*0.12751743082459617
  live=((member>>(r//32))&1)!=0;logits=tl.where(valid[None,:]&live[:,None],logits,float('-inf'))
  newmax=tl.maximum(maximum,tl.max(logits,1));safe=tl.where(newmax==float('-inf'),0.,newmax)
  alpha=tl.where(maximum==float('-inf'),0.,tl.exp2(maximum-safe));prob=tl.exp2(logits-safe[:,None])
  v=tl.load(V+(h*NK+ki[:,None])*128+d[None,:],valid[:,None],0);acc=acc*alpha[:,None];acc=tl.dot(prob.to(v.dtype),v,acc)
  denom=denom*alpha+tl.sum(prob,1);maximum=newmax
 tl.store(O+((h*N+qi[:,None])*S+split)*128+d[None,:],acc/tl.maximum(denom[:,None],1e-20),qi[:,None]<N)
 tl.store(LSE+(h*N+qi)*S+split,tl.where(denom>0,maximum+tl.log2(denom),float('-inf')),qi<N)
def attention(q,kv,union,quant,seg,ix,group=64,splits=2,warps=4,stages=2,int8=True):
 ids,mem,count=union;h,n,d=q.shape;b=kv.state['key'].shape[1];out=torch.empty_like(q)
 partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32);lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
 _att[(tr.cdiv(n,group),h,splits)](q,quant.k if int8 else kv.k,kv.v,quant.scale,ids,mem,count,partial,lse,n,kv.k.shape[1],b,group,ids.shape[1],ids.shape[2],splits,seg,int8,num_warps=warps,num_stages=stages)
 _merge_restore[(tr.cdiv(n,8),h)](partial,lse,ix,out,n,splits,8,True,num_warps=4,enable_fp_fusion=False)
 return out
