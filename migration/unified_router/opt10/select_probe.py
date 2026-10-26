import argparse,statistics,itertools
from pathlib import Path
import torch
from ..opt9.network import PackedRoute
from ..opt9.select import fused_select
from .batched_select import select
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);route=PackedRoute(batch=512);state=route.begin(d['prepared'],d['sketch']);q=d['q'][:,:512];score,_=route.score(q,state);origins=d['sketch'].origins;beta=d['beta'];ref=fused_select(score,origins,beta,'packed32');rg,_=capture(lambda:fused_select(score,origins,beta,'packed32'));rows=[]
 for group,warps in itertools.product([1,2,4,8,16],[4]):
  try:
   fn=lambda:select(score,origins,beta,group,warps);y=fn();ng,_=capture(fn);timing(rg,1000);timing(ng,1000);pairs=[]
   for i in range(6):
    if i%2:nt=timing(ng,200);rt=timing(rg,200)
    else:rt=timing(rg,200);nt=timing(ng,200)
    pairs.append((rt,nt))
   row=dict(group=group,warps=warps,reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),changed_mask_queries=int((y[0]!=ref[0]).any(-1).sum()),changed_rem_rows=int((y[1]!=ref[1]).any(-1).sum()),changed_add_rows=int((y[2]!=ref[2]).any(-1).sum()),samples=pairs)
  except Exception as e:row=dict(group=group,warps=warps,error=str(e))
  rows.append(row);atomic_json(root/'optimization10/select_probe.json',rows);print(row,flush=True)
if __name__=='__main__':main()
