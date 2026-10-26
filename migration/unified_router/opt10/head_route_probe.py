import argparse,itertools,statistics
from pathlib import Path
from types import SimpleNamespace
import torch
from ..opt9.network import PackedRoute
from .head_route import HeadRoute
from ..opt9.select import fused_select
from ..opt5.probe import capture,timing
from ..learning import ActionHead
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);q=d['q'][:,:512];sk=d['sketch'];ref=PackedRoute(batch=512);state=ref.begin(d['prepared'],sk);score,raw=ref.score(q,state);mask,rem,add=fused_select(score,sk.origins,d['beta'],'packed32');stats,cv=ref.stats(q,sk,score,mask,rem,add,raw,state)
 head=ActionHead(d['head']).cuda();head.load_state_dict(d['head']);head.eval();step=q.new_tensor(3.,dtype=torch.float32)
 rf=lambda:ref.actions(q,sk,score,mask,rem,add,d['beta'],step,stats,cv,head);yr=rf();rg,_=capture(rf);rows=[]
 for fp16,gelu,pair in itertools.product([False,True],[False,True],['tf32x3','fp16']):
  route=HeadRoute(batch=512);route.head_fp16=fp16;route.approx_gelu=gelu;route.pair_precision=pair
  fn=lambda:route.actions(q,sk,score,mask,rem,add,d['beta'],step,stats,cv,head);y=fn();ng,_=capture(fn);timing(rg,1000);timing(ng,1000);pairs=[]
  for i in range(6):
   if i%2:nt=timing(ng,200);rt=timing(rg,200)
   else:rt=timing(rg,200);nt=timing(ng,200)
   pairs.append((rt,nt))
  row=dict(fp16=fp16,gelu=gelu,pair=pair,reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),changed_queries=int((y!=yr).any(-1).sum()),samples=pairs);rows.append(row);atomic_json(root/'optimization10/head_route_probe.json',rows);print(row,flush=True)
if __name__=='__main__':main()
