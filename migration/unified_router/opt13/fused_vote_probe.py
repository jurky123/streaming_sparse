import argparse,json,statistics,time
from pathlib import Path
import torch
from ..common import config,atomic_json
from ..opt5.probe import capture,timing
from ..opt9.segmented_attention import attention
from .fused_vote import vote_union as vote
from .fused_attention import attention as fused_attention
from .setup import ROOT

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--file',default='');a=p.parse_args();root,m=config(ROOT)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 files=[root/'captured'/a.file]if a.file else sorted((root/'captured').glob('*.pt'))
 rows=[]
 for f in files:
  d=torch.load(f,map_location='cuda',weights_only=False);q=d['q'];k=d['k'];v=d['v'];mask=d['mask'];nh=d['sketch'].history_tokens
  def attend(mm):return (fused_attention if hasattr(mm,'_opt13_union')else attention)(q,k[:,:nh],v[:,:nh],k[:,nh:],v[:,nh:],mm,pack_v=True,fuse_pack=True)
  expected=attend(mask);dense=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])[0]
  rg,_=capture(lambda:attend(mask));timing(rg,100)
  pad=torch.nn.functional.pad(mask,(0,0,0,(-mask.shape[1])%32)).reshape(12,-1,32,mask.shape[-1]);counts=pad.sum(2)
  distribution={str(i):int((counts==i).sum())for i in range(33)}
  for minimum in [2,4,8]:
   mm=vote(mask,minimum);assert mm.any(-1).all()and not (mm&~mask).any()
   # Independent Torch oracle includes partial-tile threshold and orphan restoration.
   ids=torch.arange(mask.shape[1],device=q.device)//32;validrows=torch.full((counts.shape[1],),32,device=q.device);validrows[-1]=mask.shape[1]%32 or 32
   threshold=(minimum*validrows+31)//32;torch_mask=mask&(counts[:,ids]>=threshold[ids][None,:,None]);torch_mask|=mask&~torch_mask.any(-1,keepdim=True)
   assert torch.equal(mm,torch_mask)
   y=attend(mm);assert torch.equal(y,attention(q,k[:,:nh],v[:,:nh],k[:,nh:],v[:,nh:],torch_mask,pack_v=True,fuse_pack=True));ng,_=capture(lambda:attend(vote(mask,minimum)));timing(ng,100);pairs=[]
   for i in range(4):
    if i%2:nt=timing(ng,50);rt=timing(rg,50)
    else:rt=timing(rg,50);nt=timing(ng,50)
    pairs.append([rt,nt])
   c=torch.nn.functional.pad(mm,(0,0,0,(-mm.shape[1])%32)).reshape(12,-1,32,mm.shape[-1]).any(2)
   nmse=lambda a,b:float((a.float()-b.float()).square().sum()/b.float().square().sum())
   row=dict(file=f.name,minimum_votes=minimum,reference_ms=statistics.median(x[0]for x in pairs),new_ms=statistics.median(x[1]for x in pairs),reference_keep=float(mask.float().mean()),new_keep=float(mm.float().mean()),reference_union=float((counts>0).float().mean()),new_union=float(c.float().mean()),attention_nmse_vs_reference=nmse(y,expected),attention_nmse_vs_dense=nmse(y,dense),reference_nmse_vs_dense=nmse(expected,dense),vote_histogram=distribution,samples=pairs)
   rows.append(row);atomic_json(root/('fused_vote_probe_'+f.stem+'.json'),[x for x in rows if x['file']==f.name]);print(row,flush=True)
if __name__=='__main__':main()
