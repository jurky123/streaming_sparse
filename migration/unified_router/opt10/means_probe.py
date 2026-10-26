import argparse,statistics,itertools
from pathlib import Path
import torch
from ..opt9.network import PackedRoute
from ..opt9.select import fused_select
from ..opt8.dual_means import means as reference
from .split_means import means
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);route=PackedRoute(batch=512);state=route.begin(d['prepared'],d['sketch']);q=d['q'][:,:512];score,raw=route.score(q,state);mask,_,_=fused_select(score,d['sketch'].origins,d['beta'],'packed32');weights=raw[0];rf=lambda:reference(weights,mask,state,bm=32,bk=64,warps=4,stages=3);ref=rf();rg,_=capture(rf);rows=[]
 for splits,bm,bk,warps in list(itertools.product([2,4,8],[16,32,64],[32,64],[4]))[a.rank::a.world]:
  try:
   fn=lambda:means(weights,mask,state,bm=bm,bk=bk,warps=warps,stages=3,splits=splits);y=fn();ng,_=capture(fn);timing(rg,1000);timing(ng,1000);pairs=[]
   for i in range(4):
    if i%2:nt=timing(ng,200);rt=timing(rg,200)
    else:rt=timing(rg,200);nt=timing(ng,200)
    pairs.append((rt,nt))
   row=dict(splits=splits,bm=bm,bk=bk,warps=warps,reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),nmse=float((y-ref).square().sum()/ref.square().sum()),max_abs=float((y-ref).abs().max()),samples=pairs)
  except Exception as e:row=dict(splits=splits,bm=bm,bk=bk,warps=warps,error=str(e))
  rows.append(row);atomic_json(root/f'optimization10/means_probe{a.rank}.json',rows);print(row,flush=True)
if __name__=='__main__':main()
