import os,json,time,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
from local_experiments.review_v1.model import Layer
from local_experiments.review_v1.ops import SummaryOps
from local_experiments.review_v1.graph_summary import GraphSummary
from local_experiments.review_v1.split_summary import build as _build
def build(*args):return _build(*args,three_term_features=True)
r=Path('/tmp/latest_kernel_optimization4');r.mkdir(exist_ok=True);samples=torch.load('/tmp/latest_kernel_optimization2/summary_inputs.pt',map_location='cuda',weights_only=True)
def bench(fn):
 for _ in range(5):fn()
 torch.cuda.synchronize();start=time.perf_counter()
 for _ in range(50):fn()
 torch.cuda.synchronize();return (time.perf_counter()-start)*20
rows=[];graph=GraphSummary()
with torch.no_grad():
 for mode,x in samples.items():
  m=Layer(x['state']);m.load_state_dict(x['state']);k=x['k'];v=x['v'];cheap=mode=='cheap'
  for change in [False,True]:
   if change:m.pool_k.add_(.03);m.group_gate.add_(.02);v=v*.99
   ref=SummaryOps.build(m,k,v,cheap);out=build(m,k,v,cheap);errors={}
   for n in ['keys','mean','detail','stats']:
    a=getattr(out,n).float();b=getattr(ref,n).float();delta=a-b;errors[n]=dict(nmse=float(delta.square().mean()/b.square().mean().clamp_min(1e-12)),max_abs=float(delta.abs().max()),equal_fraction=float((a==b).float().mean()),finite=bool(a.isfinite().all()))
   row=dict(mode=mode,changed=change,errors=errors,previous_ms=bench(lambda:graph(m,k,v,cheap)),fused_ms=bench(lambda:build(m,k,v,cheap)));rows.append(row);print(row,flush=True)
   assert all(v['finite'] and v['nmse']<2e-7 for v in errors.values()),row
 (r/'three_term_summary_checks.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
