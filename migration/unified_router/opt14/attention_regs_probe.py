import argparse,json,itertools,statistics,os
import torch
from ..common import atomic_json
from ..opt9.segmented_attention import attention as reference
from ..opt5.probe import capture,timing
from .attention_regs import attention
from .setup import ROOT,PARENT
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--file',default='layer24_chunk3.pt');p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);p.add_argument('--selected',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);d=torch.load(PARENT/'captured'/a.file,map_location='cuda',weights_only=False);q,k,v,mask=[d[n]for n in ['q','k','v','mask']];nh=d['sketch'].history_tokens;args=[q,k[:,:nh],v[:,:nh],k[:,nh:],v[:,nh:],mask]
 ref=lambda:reference(*args,pack_v=True,fuse_pack=True);expected=ref();rg,_=capture(ref);timing(rg,100);rows=[]
 configs=list(itertools.product([32],[2],[80,96,112,128,160,192,224,256],[2,4],[2]))
 if a.selected:configs=[(16,2,2,4,2),(16,2,2,2,2),(32,2,2,4,2),(32,2,2,2,2),(32,2,1,2,2),(16,2,1,2,2)]
 for group,splits,pages,warps,stages in configs[a.rank::a.world]:
  row=dict(group=group,splits=splits,pack_v=True,fuse_pack=True,maxnreg=pages,warps=warps,stages=stages)
  try:
   fn=lambda:attention(*args,group=group,splits=splits,pack_v=True,fuse_pack=True,maxnreg=pages,warps=warps,stages=stages)
   y=fn();assert torch.isfinite(y).all();ng,_=capture(fn);timing(ng,100);pairs=[]
   for i in range(4):
    if i%2:nt=timing(ng,40);rt=timing(rg,40)
    else:rt=timing(rg,40);nt=timing(ng,40)
    pairs.append([rt,nt])
   row.update(reference_ms=statistics.median(x[0]for x in pairs),new_ms=statistics.median(x[1]for x in pairs),nmse=float((y.float()-expected.float()).square().sum()/expected.float().square().sum()),max_abs=float((y.float()-expected.float()).abs().max()),bit_identical=torch.equal(y,expected),samples=pairs)
  except Exception as e:row['error']=str(e)
  rows.append(row);print(row,flush=True);atomic_json(ROOT/f'attention_regs_{a.file.removesuffix(".pt")}_rank{a.rank}.json',rows)
if __name__=='__main__':main()
