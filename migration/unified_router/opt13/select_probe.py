import json,statistics,torch
from pathlib import Path
from ..common import config,atomic_json
from ..opt10.compact_route import CompactMomentRoute
from ..opt9.select import fused_select
from ..opt5.probe import capture,timing
from .select import select
from .setup import ROOT
@torch.no_grad()
def main():
 root,m=config(ROOT);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(Path(m['parent_experiment'])/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
 route=CompactMomentRoute(batch=4680);state=route.begin(d['prepared'],d['sketch']);score,_=route.score(d['q'],state);rows=[]
 # Include all history lengths, full query set, and partial final query tile.
 for b in [148,222,296]:
  x=score[...,:b].contiguous();o=d['sketch'].origins.clone()[:b];o[-74:]=1.;o[:-74]=0.
  expected=fused_select(x,o,d['beta'],'packed32');rg,_=capture(lambda:fused_select(x,o,d['beta'],'packed32'));timing(rg,100)
  for warps in [4,8,2,1]:
   try:
    fn=lambda:select(x,o,d['beta'],warps);actual=fn();ng,_=capture(fn);timing(ng,100);pairs=[]
    for i in range(4):
     if i%2:nt=timing(ng,80);rt=timing(rg,80)
     else:rt=timing(rg,80);nt=timing(ng,80)
     pairs.append([rt,nt])
    row=dict(pages=b,warps=warps,reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),changed=[int((a!=e).any(-1).sum())for a,e in zip(actual,expected)],samples=pairs)
   except Exception as error:row=dict(pages=b,warps=warps,error=str(error))
   rows.append(row);atomic_json(root/'select_probe.json',rows);print(row,flush=True)
if __name__=='__main__':main()
