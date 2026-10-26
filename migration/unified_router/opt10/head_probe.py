import argparse,statistics,itertools
from pathlib import Path
import torch,triton as tr
from ..opt9.network import PackedRoute
from ..opt9.select import fused_select
from ..opt6.factor import _factor_prepare,_factor_finish
from .batched_finish import _finish
from ..opt5.probe import capture,timing
from ..learning import ActionHead
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);q=data['q'][:,:512];sk=data['sketch'];route=PackedRoute(batch=512);state=route.begin(data['prepared'],sk);score,raw=route.score(q,state);mask,rem,add=fused_select(score,sk.origins,data['beta'],'packed32');stats,cv=route.stats(q,sk,score,mask,rem,add,raw,state);h,n,b=score.shape;step=q.new_tensor(3.,dtype=torch.float32)
 head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval();tmp=torch.empty((h*n,384),device=q.device)
 _factor_prepare[(h*n,)](stats,score,cv,sk.costs,sk.origins,rem,add,head.w1,head.b1,head.mean,head.std,step,tmp,n,b,sk.history_tokens,sk.current_tokens,data['beta'],'tf32x3',num_warps=1,enable_fp_fusion=False)
 def old():
  out=torch.empty_like(mask);_factor_finish[(h*n,)](tmp,stats,mask,rem,add,head.w1,head.w2,head.b2,out,n,b,tr.next_power_of_2(b),num_warps=1,enable_fp_fusion=False);return out
 ref=old();rg,_=capture(old);rows=[]
 for g,warps,approx in itertools.product([1,2,4,8],[4],[False,True]):
  try:
   def fn():
    out=torch.empty_like(mask);_finish[(tr.cdiv(n,g),h)](tmp,stats,mask,rem,add,head.w1,head.w2,head.b2,out,n,b,tr.next_power_of_2(b),g,approx,num_warps=warps,enable_fp_fusion=False);return out
   y=fn();ng,_=capture(fn);timing(rg,1000);timing(ng,1000);pairs=[]
   for i in range(4):
    if i%2:nt=timing(ng,200);rt=timing(rg,200)
    else:rt=timing(rg,200);nt=timing(ng,200)
    pairs.append((rt,nt))
   row=dict(group=g,warps=warps,approx=approx,reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),changed_queries=int((y!=ref).any(-1).sum()),samples=pairs)
  except Exception as e:row=dict(group=g,warps=warps,approx=approx,error=str(e))
  rows.append(row);atomic_json(root/'optimization10/head_probe.json',rows);print(row,flush=True)
if __name__=='__main__':main()
