import os,json,time,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
from local_experiments.review_v1.model import Layer
from local_experiments.review_v1.ops import SummaryOps
from local_experiments.review_v1.graph_summary import GraphSummary
from local_experiments.review_v1.fused_summary import build
r=Path('/tmp/latest_kernel_optimization4');r.mkdir(exist_ok=True);samples=torch.load('/tmp/latest_kernel_optimization2/summary_inputs.pt',map_location='cuda',weights_only=True)
def bench(fn):
 for _ in range(5):fn()
 torch.cuda.synchronize();start=time.perf_counter()
 for _ in range(50):fn()
 torch.cuda.synchronize();return (time.perf_counter()-start)*20

rows=[]
with torch.no_grad():
 for mode,x in samples.items():
  m=Layer(x['state']);m.load_state_dict(x['state']);k=x['k'];v=x['v'];cheap=mode=='cheap';ref=SummaryOps.build(m,k,v,cheap)
  for pw,sw in [(4,4),(8,4),(4,8),(8,8),(4,16),(8,16)]:
   actual=build(m,k,v,cheap,pool_warps=pw,summary_warps=sw)
   errors={n:float((getattr(actual,n).float()-getattr(ref,n).float()).square().mean()/getattr(ref,n).float().square().mean().clamp_min(1e-12)) for n in ['keys','mean','detail','stats']}
   ms=bench(lambda:build(m,k,v,cheap,pool_warps=pw,summary_warps=sw));row=dict(mode=mode,pool_warps=pw,summary_warps=sw,nmse=errors,ms=ms);rows.append(row);print(row,flush=True);assert max(errors.values())<2e-7
 (r/'summary_launch_tuning.json').write_text(json.dumps(rows,indent=2))
