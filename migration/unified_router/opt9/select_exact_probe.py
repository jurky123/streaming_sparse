import argparse,json,statistics
from pathlib import Path
import torch
from ..opt8.projected_route import ProjectedRoute
from ..opt5.select import fused_select as reference
from .select import fused_select
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);out=root/'optimization9';out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);route=ProjectedRoute(batch=512);sk=data['sketch'];state=route.begin(data['prepared'],sk);rows=[]
 for n in [512,72]:
  score,_=route.score(data['q'][:,-n:],state);origins=sk.origins;beta=data['beta']
  for case in ['actual','ties','constant','nohistory','onecurrent']:
   x=score if case in ['actual','nohistory','onecurrent']else score.round()if case=='ties'else torch.zeros_like(score)
   o=torch.ones_like(origins)if case=='nohistory'else origins.clone()
   if case=='onecurrent':o.zero_();o[-1]=1
   ref=reference(x,o,beta)
   for mode in ['exact32']:
    actual=fused_select(x,o,beta,mode)
    row=dict(n=n,case=case,mode=mode,changed_mask=int((actual[0]!=ref[0]).sum()),changed_remove_rows=int((actual[1]!=ref[1]).any(-1).sum()),changed_add_rows=int((actual[2]!=ref[2]).any(-1).sum()))
    if mode!='packed32':assert all(torch.equal(u,v)for u,v in zip(ref,actual)),row
    if case=='actual':
     rg,_=capture(lambda:reference(x,o,beta));ng,_=capture(lambda:fused_select(x,o,beta,mode));timing(rg,200);timing(ng,200);pairs=[]
     for i in range(6):
      if i%2:nt=timing(ng,200);rt=timing(rg,200)
      else:rt=timing(rg,200);nt=timing(ng,200)
      pairs.append((rt,nt))
     row.update(reference_ms=statistics.median(r[0]for r in pairs),new_ms=statistics.median(r[1]for r in pairs),samples=pairs)
    rows.append(row);atomic_json(out/'select_exact32_probe.json',rows);print(row,flush=True)
if __name__=='__main__':main()
