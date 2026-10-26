import argparse,statistics
from pathlib import Path
import torch
from ..common import atomic_json
from ..opt8.block_int8_attention import attention as reference
from .segmented_attention import attention
from ..opt5.probe import capture,timing
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);q=data['q'];k=data['k'].transpose(0,1).contiguous().transpose(0,1);v=data['v'].transpose(0,1).contiguous().transpose(0,1);mask=data['mask'];nh=k.shape[1]-4680;kh=k[:,:nh];kc=k[:,nh:];vh=v[:,:nh];vc=v[:,nh:]
 ref=lambda:reference(q,torch.cat((kh,kc),1),torch.cat((vh,vc),1),mask,warps=2)
 fn=lambda:attention(q,kh,vh,kc,vc,mask,pack_v=True);yr=ref();y=fn();assert torch.isfinite(y).all();rg,_=capture(ref);ng,_=capture(fn);timing(rg,300);timing(ng,300);pairs=[]
 for i in range(6):
  if i%2:nt=timing(ng,100);rt=timing(rg,100)
  else:rt=timing(rg,100);nt=timing(ng,100)
  pairs.append((rt,nt))
 row=dict(reference_ms=statistics.median(r[0]for r in pairs),new_ms=statistics.median(r[1]for r in pairs),nmse=float((y.float()-yr.float()).square().sum()/yr.float().square().sum()),max_abs=float((y.float()-yr.float()).abs().max()),bit_identical=torch.equal(y,yr),samples=pairs,includes_original_KV_cat=True,native_strides=list(k.stride()))
 atomic_json(root/'optimization9/memory_packv_probe.json',row);print(row,flush=True)
if __name__=='__main__':main()
