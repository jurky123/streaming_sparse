"""Isolated A100 operator profile, diagnostic only (not end-to-end ranking)."""
import os,time,json,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(123)
from local_experiments.review_v1.model import Model
from local_experiments.review_v1.ops import SummaryOps,RouteOps,Config
from local_experiments.review_v1.runtime import SelectedAttention
from local_experiments.review_v1.fast_attention import attention
from local_experiments.review_v1.fast_route import GraphRoute
@torch.no_grad()
def main():
 m=Model('artifacts/four_gaps_20261031/shared/epoch1.pt');m.load_state_dict(torch.load('artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt',weights_only=False)['model']);m=m[1]
 k=torch.randn(12,432,64,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);q=torch.randn(12,4608,128,device='cuda',dtype=torch.bfloat16);s=SummaryOps.build(m,k,v)
 probes=q[:,:128].contiguous();g=GraphRoute()
 def route_ref():
  f,_=RouteOps.features(probes,s);return (f*m.theta[:,None,None]).sum(-1).reshape(12,8,16,432).mean(2)
 scores=route_ref().repeat(1,9,1);own=(torch.arange(72,device='cuda')+360)[None].expand(12,-1);selection=RouteOps.select(scores,own,Config(history_pages=360));selection['fixed_count']=43
 def measure(fn):
  fn();torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(3):fn()
  torch.cuda.synchronize();return (time.perf_counter()-start)*1000/3
 result={}
 for name,fn in [('route_8_tiles_reference',route_ref),('route_8_tiles_graph',lambda:g(probes,s,m.theta)),('attention_72_tiles_reference',lambda:SelectedAttention.run(q,k,v,s,m.basis,selection)),('attention_72_tiles_fast',lambda:attention(q,k,v,s,m.basis,selection))]:
  result[name]=measure(fn);print(name,result[name],flush=True)
 Path('/tmp/latest_operator_profile.json').write_text(json.dumps(result,indent=2))
main()
