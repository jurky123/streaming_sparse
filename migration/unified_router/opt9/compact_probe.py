import argparse,statistics
from pathlib import Path
import torch
from ..opt8.projected_route import ProjectedRoute
from .compact_energy import score
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);route=ProjectedRoute(batch=512);state=route.begin(data['prepared'],data['sketch']);q=data['q'][:,:512];ref=route.score(q,state);rg,_=capture(lambda:route.score(q,state));rows=[]
 for qm in [32,64]:
  for width in [16,32,64]:
   try:
    fn=lambda:score(q,state,qm,width);y=fn();ng,_=capture(fn);timing(ng,100);timing(rg,100);pairs=[]
    for i in range(6):
     if i%2:nt=timing(ng,100);rt=timing(rg,100)
     else:rt=timing(rg,100);nt=timing(ng,100)
     pairs.append((rt,nt))
    nmse=lambda x,y:float((x.float()-y.float()).square().sum()/y.float().square().sum())
    row=dict(qm=qm,width=width,reference_ms=statistics.median(r[0]for r in pairs),new_ms=statistics.median(r[1]for r in pairs),score_nmse=nmse(y[0],ref[0]),weights_nmse=nmse(y[1][0],ref[1][0]),energy_nmse=nmse(y[1][1],ref[1][1]),samples=pairs)
   except Exception as e:row=dict(qm=qm,width=width,error=str(e))
   rows.append(row);atomic_json(root/'optimization9/compact_probe.json',rows);print(row,flush=True)
if __name__=='__main__':main()
