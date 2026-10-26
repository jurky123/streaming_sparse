"""Primary-library baseline: PyTorch FlexAttention with exact per-query masks.
Includes KV page padding, union construction and BlockMask transpose metadata.
"""
import json,statistics,torch,triton as tr
import triton.language as tl
from torch.nn.attention.flex_attention import flex_attention,BlockMask
from ..fast_kernels import _union
from ..opt9.segmented_attention import attention as reference
from ..opt5.probe import capture,timing
from ..common import atomic_json
from .setup import ROOT,PARENT
@tr.jit
def _pack(KH,KC,VH,VC,KO,VO,B:tl.constexpr,NH:tl.constexpr,KHS:tl.constexpr,KHT:tl.constexpr,KCS:tl.constexpr,KCT:tl.constexpr,VHS:tl.constexpr,VHT:tl.constexpr,VCS:tl.constexpr,VCT:tl.constexpr):
 page=tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,128);ki=page//74*4680+page%74*64+t;valid=page%74*64+t<4680
 kp=tl.where(ki[:,None]<NH,KH+h*KHS+ki[:,None]*KHT+d[None,:],KC+h*KCS+(ki[:,None]-NH)*KCT+d[None,:]);vp=tl.where(ki[:,None]<NH,VH+h*VHS+ki[:,None]*VHT+d[None,:],VC+h*VCS+(ki[:,None]-NH)*VCT+d[None,:])
 k=tl.load(kp,valid[:,None],0);v=tl.load(vp,valid[:,None],0);dst=((h*B+page)*64+t[:,None])*128+d[None,:];tl.store(KO+dst,k);tl.store(VO+dst,v)
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(PARENT/'captured/layer24_chunk3.pt',map_location='cuda',weights_only=False);q,k,v,mask=[d[n]for n in ['q','k','v','mask']];nh=d['sketch'].history_tokens;kh,kc=k[:,:nh],k[:,nh:];vh,vc=v[:,:nh],v[:,nh:];h,n,dim=q.shape;b=mask.shape[-1];rows=[]
 ref=lambda:reference(q,kh,vh,kc,vc,mask,pack_v=True,fuse_pack=True);expected=ref();rg,_=capture(ref);timing(rg,100)
 # Compile the library kernel once per tile configuration; BlockMask metadata
 # remains dynamic inputs. q_num_blocks is constructed correctly by the API.
 for group in [32,64]:
  opts={'BLOCK_M':group,'BLOCK_N':64,'num_warps':4,'num_stages':2}
  compiled=torch.compile(flex_attention,dynamic=False)
  def fn():
   ko=torch.empty((h,b*64,128),device=q.device,dtype=q.dtype);vo=torch.empty_like(ko)
   _pack[(b,h)](kh,kc,vh,vc,ko,vo,b,nh,kh.stride(0),kh.stride(1),kc.stride(0),kc.stride(1),vh.stride(0),vh.stride(1),vc.stride(0),vc.stride(1),num_warps=4)
   ids=torch.empty((h,tr.cdiv(n,group),b),device=q.device,dtype=torch.int32);counts=torch.empty(ids.shape[:2],device=q.device,dtype=torch.int32)
   _union[(tr.cdiv(n,group),h)](mask,ids,counts,n,b,group,tr.next_power_of_2(b),num_warps=4)
   def visible(batch,head,qi,ki):return mask[head,qi.clamp(max=n-1),ki//64]&((ki//64%74!=73)|(ki%64<8))
   bm=BlockMask.from_kv_blocks(counts[None],ids[None],BLOCK_SIZE=(group,64),mask_mod=visible,seq_lengths=(n,b*64))
   return compiled(q[None],ko[None],vo[None],block_mask=bm,kernel_options=opts)[0]
  try:
   y=fn();assert torch.isfinite(y).all();ng,_=capture(fn);timing(ng,100);pairs=[]
   for i in range(4):
    if i%2:nt=timing(ng,30);rt=timing(rg,30)
    else:rt=timing(rg,30);nt=timing(ng,30)
    pairs.append([rt,nt])
   row=dict(group=group,reference_ms=statistics.median(x[0]for x in pairs),new_ms=statistics.median(x[1]for x in pairs),nmse_vs_int8_compact=float((y.float()-expected.float()).square().sum()/expected.float().square().sum()),samples=pairs)
  except Exception as e:
   import traceback;traceback.print_exc();row=dict(group=group,error=str(e))
  rows.append(row);print(row,flush=True);atomic_json(ROOT/'flex_probe.json',rows)
if __name__=='__main__':main()
