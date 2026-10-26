"""Grouped sparse attention with wider KV iterations and fixed Q quantization.
G16 and G32 both quantize Q with original 32-query scales. All query masks
are unchanged; each iteration can visit 1/2/4 compacted KV pages together.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint
from ..fast_kernels import _union
from ..opt9.segmented_attention import _quantize,_merge

@tr.jit
def _split(QP,KP,VP,KS,M,IDS,COUNTS,OUT,LSE,
 Q:tl.constexpr,K:tl.constexpr,B:tl.constexpr,QH:tl.constexpr,QQ:tl.constexpr,
 G:tl.constexpr,SPLITS:tl.constexpr,PAGES:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);split=tl.program_id(2)
 qi=g*G+tl.arange(0,G);d=tl.arange(0,128);t=tl.arange(0,64*PAGES)
 if G==16:
  q32=g//2*32+tl.arange(0,32)
  qq=tl.load(QP+h*QH+q32[:,None]*QQ+d[None,:],q32[:,None]<Q,0)
  qs=tl.maximum(tl.max(tl.max(tl.abs(qq.to(tl.float32)),1),0)/127.,1.e-10)
  q=tl.load(QP+h*QH+qi[:,None]*QQ+d[None,:],qi[:,None]<Q,0)
 else:
  q=tl.load(QP+h*QH+qi[:,None]*QQ+d[None,:],qi[:,None]<Q,0)
  qs=tl.maximum(tl.max(tl.max(tl.abs(q.to(tl.float32)),1),0)/127.,1.e-10)
 q=nearbyint(q.to(tl.float32)/qs).to(tl.int8)
 maximum=tl.full((G,),float('-inf'),tl.float32);denom=tl.zeros((G,),tl.float32);acc=tl.zeros((G,128),tl.float32)
 count=tl.load(COUNTS+h*tr.cdiv(Q,G)+g);part=tl.cdiv(count,SPLITS);end=tl.minimum((split+1)*part,count)
 for j in range(split*part,end,PAGES):
  page=tl.load(IDS+(h*tr.cdiv(Q,G)+g)*B+j+t//64,j+t//64<end,0)
  local=page%74;ki=page//74*4680+local*64+t%64
  valid=(j+t//64<end)&(ki<K)&(local*64+t%64<4680)
  k=tl.load(KP+(h*K+ki[None,:])*128+d[:,None],valid[None,:],0)
  ks=tl.load(KS+h*B+page)
  logits=tl.dot(q,k).to(tl.float32)*(qs*ks[None,:])*0.12751743082459617
  visible=tl.load(M+(h*Q+qi[:,None])*B+page[None,:],qi[:,None]<Q,False)
  logits=tl.where(visible&valid[None,:],logits,float('-inf'))
  newmax=tl.maximum(maximum,tl.max(logits,1));safe=tl.where(newmax==float('-inf'),0.,newmax)
  alpha=tl.exp2(maximum-safe);prob=tl.exp2(logits-safe[:,None])
  v=tl.load(VP+(h*K+ki[:,None])*128+d[None,:],valid[:,None],0)
  acc=acc*alpha[:,None]+tl.dot(prob.to(v.dtype),v)
  denom=denom*alpha+tl.sum(prob,1);maximum=newmax
 normal=tl.where(denom>0,denom,1.)
 tl.store(OUT+((h*Q+qi[:,None])*SPLITS+split)*128+d[None,:],acc/normal[:,None],qi[:,None]<Q)
 tl.store(LSE+(h*Q+qi)*SPLITS+split,maximum+tl.log2(normal),qi<Q)

def attention(q,kh,vh,kc,vc,mask,group=32,splits=2,warps=4,stages=2,pages=2):
 assert group in [16,32] and splits in [1,2,4,8] and pages in [1,2,4]
 h,n,d=q.shape;b=mask.shape[-1];nh=kh.shape[1];nk=nh+kc.shape[1]
 vp=torch.empty((h,nk,d),device=q.device,dtype=vh.dtype);ki=torch.empty((h,nk,d),device=q.device,dtype=torch.int8);ks=torch.empty((h,b),device=q.device,dtype=torch.float32)
 _quantize[(b,h)](kh,kc,ki,ks,vh,vc,vp,nk,b,kh.stride(0),kh.stride(1),kc.stride(0),kc.stride(1),nh,vh.stride(0),vh.stride(1),vc.stride(0),vc.stride(1),True,num_warps=4)
 ids=torch.empty((h,tr.cdiv(n,group),b),device=q.device,dtype=torch.int32);counts=torch.empty(ids.shape[:2],device=q.device,dtype=torch.int32)
 _union[(tr.cdiv(n,group),h)](mask,ids,counts,n,b,group,tr.next_power_of_2(b),num_warps=4)
 out=torch.empty_like(q,memory_format=torch.contiguous_format);partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32);lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
 _split[(tr.cdiv(n,group),h,splits)](q,ki,vp,ks,mask,ids,counts,partial,lse,n,nk,b,q.stride(0),q.stride(1),group,splits,pages,num_warps=warps,num_stages=stages)
 _merge[(h*n,)](partial,lse,out,n,splits,num_warps=4)
 return out
