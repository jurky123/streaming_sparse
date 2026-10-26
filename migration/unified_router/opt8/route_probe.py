"""Tune the full router's actual candidate selection and masked-mean kernels."""
import argparse,itertools,os,json
from pathlib import Path
import torch,triton as tr
from ..opt5.probe import capture,timing
from ..opt6_guarded import GuardedHeadRoute
from ..opt5.select import _fused_select,fused_select
from ..opt4_fused import _masked_mean,means
from ..common import atomic_json
from .dual_means import means as dual_mean
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--target',choices=['select','means','dual'],required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);a=p.parse_args();root=Path(a.directory);out=root/'optimization8';torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);q=data['q'][:,:512];route=GuardedHeadRoute(batch=512);sk=data['sketch'];state=route.begin(data['prepared'],sk);score,raw=route.score(q,state);mask,rem,add=fused_select(score,sk.origins,data['beta']);w,energy=raw;h,n,b=mask.shape;r=w.shape[1]
 def select(warps):
  s=torch.empty_like(mask);rm=torch.empty_like(rem);ad=torch.empty_like(add)
  _fused_select[(h*n,)](score,s,sk.origins,rm,ad,n,b,tr.next_power_of_2(b),data['beta'],num_warps=warps,enable_fp_fusion=False)
  return s,rm,ad
 def mean(bm,bk,warps,stages):
  out=torch.empty((h,2*n,32),device=q.device,dtype=torch.float32)
  _masked_mean[(tr.cdiv(2*n,bm),h)](w,state['value'],mask,state['page_index'],out,n,b,r,bm,bk,'fp16',num_warps=warps,num_stages=stages)
  return out
 ref_fn=(lambda:select(4))if a.target=='select'else(lambda:mean(64,32,4,2));ref=ref_fn();rg,_=capture(ref_fn)
 configs=[(x,)for x in [1,2,4,8]]if a.target=='select'else list(itertools.product([32,64,128],[32,64,128],[4,8],[2,3]))
 rows=[]
 for cfg in configs[a.rank::a.world]:
  try:
   fn=(lambda:select(*cfg))if a.target=='select'else(lambda:dual_mean(w,mask,state,*cfg))if a.target=='dual'else(lambda:mean(*cfg));actual=fn();ng,_=capture(fn);timing(ng,100);timing(rg,100);samples=[]
   for i in range(4):
    if i%2:nt=timing(ng,100);rt=timing(rg,100)
    else:rt=timing(rg,100);nt=timing(ng,100)
    samples.append((rt,nt))
   row=dict(config=cfg,reference_ms=sum(x[0]for x in samples)/4,new_ms=sum(x[1]for x in samples)/4)
   if a.target=='select':row.update(changed_mask_queries=int((actual[0]!=ref[0]).any(-1).sum()),changed_remove_rows=int((actual[1]!=ref[1]).any(-1).sum()),changed_add_rows=int((actual[2]!=ref[2]).any(-1).sum()))
   else:row.update(nmse=float((actual-ref).square().sum()/ref.square().sum()),max_abs=float((actual-ref).abs().max()),identical=torch.equal(actual,ref))
  except Exception as e:row=dict(config=cfg,error=str(e))
  rows.append(row);atomic_json(out/f'{a.target}_tune{a.rank}.json',rows);print(row,flush=True)
if __name__=='__main__':main()
