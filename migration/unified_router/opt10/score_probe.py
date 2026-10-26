import argparse,itertools,statistics
from pathlib import Path
import torch,triton as tr
from ..opt9.network import PackedRoute
from ..opt8.projection import project
from ..opt5.kernels import _score_energy
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);route=PackedRoute(batch=512);state=route.begin(d['prepared'],d['sketch']);q=d['q'][:,:512];h,n,_=q.shape;b=state['costs'].numel();r=state['value'].shape[1];qq=project(q,state['prepared'][0]);rows=[]
 def tensors():
  score=torch.empty((h,n,b),device=q.device,dtype=torch.float32);return score,torch.empty_like(score),torch.empty((h,r,n),device=q.device,dtype=torch.float16)
 def fn(cfg,out):
  qm,gp,warps=cfg;score,energy,weights=out
  return _score_energy[(tr.cdiv(n,qm),tr.cdiv(b,gp),h)](qq,state['padded_key'],state['padded_lc'],state['grams'],state['padded_value'],score,energy,n,b,qm,gp,True,weights,state['starts'],state['counts'],state['costs'],r,True,num_warps=warps,num_stages=1,enable_fp_fusion=True)
 ref=tensors();fn((64,4,4),ref);rg,_=capture(lambda:fn((64,4,4),ref));timing(rg,1000)
 for cfg in list(itertools.product([16,32,64,128],[2,4,8],[4,8]))[a.rank::a.world]:
  try:
   out=tensors();kernel=fn(cfg,out);ng,_=capture(lambda:fn(cfg,out));timing(ng,500);pairs=[]
   for i in range(4):
    if i%2:nt=timing(ng,200);rt=timing(rg,200)
    else:rt=timing(rg,200);nt=timing(ng,200)
    pairs.append((rt,nt))
   nmse=lambda x,y:float((x.float()-y.float()).square().sum()/y.float().square().sum())
   row=dict(qm=cfg[0],gp=cfg[1],warps=cfg[2],reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),score_nmse=nmse(out[0],ref[0]),energy_nmse=nmse(out[1],ref[1]),weights_nmse=nmse(out[2],ref[2]),registers=kernel.n_regs,spills=kernel.n_spills,shared=kernel.metadata.shared,samples=pairs)
  except Exception as e:row=dict(config=cfg,error=str(e))
  rows.append(row);atomic_json(root/f'optimization10/score_probe{a.rank}.json',rows);print(row,flush=True)
if __name__=='__main__':main()
