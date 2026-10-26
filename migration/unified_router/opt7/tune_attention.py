import argparse,itertools,os
from pathlib import Path
import torch
from ..opt2.attention import attention as old
from .split_attention import attention as split
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory)
 data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
 args=[data[x]for x in ['q','k','v','mask']];ref=old(*args,group=32,warps=4,stages=3)
 rg,_=capture(lambda:old(*args,group=32,warps=4,stages=3))
 configs=[(g,w,s,sp)for g,w,s,sp in itertools.product([16,32,64,128],[2,4],[2,3,4],[1,2])]
 rows=[];rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
 for g,w,s,sp in configs[rank::world]:
  try:
   fn=lambda:(old(*args,group=g,warps=w,stages=s)if sp==1 else split(*args,group=g,warps=w,stages=s,splits=sp))
   y=fn();ng,_=capture(fn);timing(ng,100)
   samples=[]
   for i in range(4):
    if i%2:nt=timing(ng,30);rt=timing(rg,30)
    else:rt=timing(rg,30);nt=timing(ng,30)
    samples.append((rt,nt))
   row=dict(group=g,warps=w,stages=s,splits=sp,reference_ms=sum(r for r,n in samples)/4,new_ms=sum(n for r,n in samples)/4,nmse=float((y.float()-ref.float()).square().sum()/ref.float().square().sum()))
  except Exception as e:row=dict(group=g,warps=w,stages=s,splits=sp,error=str(e))
  rows.append(row);print(row,flush=True);atomic_json(root/f'optimization7/tune{rank}.json',rows)
if __name__=='__main__':main()
