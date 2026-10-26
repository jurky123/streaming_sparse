import argparse,statistics
from pathlib import Path
import torch,triton as tr
from ..opt9.network import PackedRoute
from ..opt9.select import _packed_select,fused_select
from ..opt8.dual_means import means
from ..opt4.kernels import _gram_stats
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);route=PackedRoute(batch=512);state=route.begin(d['prepared'],d['sketch']);q=d['q'][:,:512];score,raw=route.score(q,state);mask,rem,add=fused_select(score,d['sketch'].origins,d['beta'],'packed32');h,n,b=score.shape;sk=d['sketch'];mean=means(raw[0],mask,state,bm=32,bk=64,warps=4,stages=3);rows=[]
 def fn(target,warps):
  if target=='select':
   mm=torch.empty_like(mask);rr=torch.empty_like(rem);aa=torch.empty_like(add);_packed_select[(h*n,)](score,mm,sk.origins,rr,aa,n,b,tr.next_power_of_2(b),d['beta'],'packed32',num_warps=warps,enable_fp_fusion=False);return mm,rr,aa
  st=torch.empty((h*n,80),device=q.device);_gram_stats[(h*n,)](mean,raw[1],score,mask,sk.costs,sk.origins,st,n,b,tr.next_power_of_2(b),num_warps=warps,enable_fp_fusion=False);return st[:,:73],
 for target in ['select','stats']:
  ref=fn(target,4);rg,_=capture(lambda:fn(target,4));timing(rg,1000)
  for warps in [1,2,4,8]:
   actual=fn(target,warps);ng,_=capture(lambda:fn(target,warps));timing(ng,1000);pairs=[]
   for i in range(4):
    if i%2:nt=timing(ng,200);rt=timing(rg,200)
    else:rt=timing(rg,200);nt=timing(ng,200)
    pairs.append((rt,nt))
   row=dict(target=target,warps=warps,reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),exact=all(torch.equal(x,y)for x,y in zip(ref,actual)),changed_rows=[int((x!=y).any(-1).sum())for x,y in zip(ref,actual)],samples=pairs)
   rows.append(row);atomic_json(root/'optimization10/reduction_probe.json',rows);print(row,flush=True)
if __name__=='__main__':main()
