import argparse,statistics
from pathlib import Path
import torch
from ..common import atomic_json
from ..opt8.block_int8_attention import attention as reference
from ..opt7.split_attention import attention as bf16
from ..opt5.probe import capture,timing
from .skip_attention import attention
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);out=root/'optimization9';torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);q=data['q'];k=data['k'];v=data['v'];mask=data['mask'];ref=lambda:reference(q,k,v,mask,warps=2);yr=ref();yb=bf16(q,k,v,mask);rg,_=capture(ref);rows=[]
 for threshold in [-100.,-12.,-8.,-4.,-2.,0.]:
  fn=lambda:attention(q,k,v,mask,warps=2,threshold=threshold);y=fn();assert torch.isfinite(y).all();ng,_=capture(fn);timing(ng,100);timing(rg,100);pairs=[]
  for i in range(6):
   if i%2:nt=timing(ng,50);rt=timing(rg,50)
   else:rt=timing(rg,50);nt=timing(ng,50)
   pairs.append((rt,nt))
  nmse=lambda x,y:float((x.float()-y.float()).square().sum()/y.float().square().sum())
  row=dict(threshold_log2=threshold,reference_ms=statistics.median(r[0]for r in pairs),new_ms=statistics.median(r[1]for r in pairs),nmse_vs_opt8=nmse(y,yr),nmse_vs_bf16=nmse(y,yb),samples=pairs);rows.append(row);atomic_json(out/'skip_probe.json',rows);print(row,flush=True)
if __name__=='__main__':main()
