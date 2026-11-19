"""Validate same risk scores/selections and profile real steady-state input."""
import os,json,time,torch
from pathlib import Path
from torch.profiler import profile,ProfilerActivity
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
from local_experiments.review_v1.ops import Summary,RouteOps,Config
from local_experiments.review_v1.fast_route import GraphRoute
from local_experiments.review_v1.fused_route import FusedRiskRoute
out=Path('/tmp/latest_kernel_optimization');sample={k:v.cuda() for k,v in torch.load(out/'route_inputs.pt',weights_only=True).items()}
rows=[];fused=FusedRiskRoute();graph=GraphRoute()
with torch.no_grad():
 for p in [216,288,360,432]:
  q=sample['q'];s=Summary(*(sample[k][:,:p].contiguous() for k in ['keys','mean','detail','stats']));theta=sample['theta']
  f,_=RouteOps.features(q,s);ref=(f*theta[:,None,None]).sum(-1).reshape(12,8,16,p).mean(2);actual=fused(q,s,theta)
  diff=actual-ref;nmse=float(diff.square().mean()/ref.square().mean());relmax=float(diff.abs().max()/ref.abs().max())
  own=(torch.arange(8,device='cuda')+p-72)[None].expand(12,-1);a=RouteOps.select(actual,own,Config(history_pages=360));b=RouteOps.select(ref,own,Config(history_pages=360));overlap=float((a['mask']&b['mask']).sum()/b['mask'].sum())
  row=dict(pages=p,score_nmse=nmse,relative_max_error=relmax,selected_overlap=overlap,bitwise_equal=torch.equal(actual,ref));rows.append(row);print(row,flush=True)
  assert torch.isfinite(actual).all() and nmse<1e-8 and relmax<5e-5 and overlap>=.999,row
 s=Summary(*(sample[k] for k in ['keys','mean','detail','stats']));q=sample['q'];theta=sample['theta']
 def bench(fn):
  for _ in range(5):fn()
  torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(50):fn()
  torch.cuda.synchronize();return (time.perf_counter()-start)*1000/50
 times=dict(graph_ms=bench(lambda:graph(q,s,theta)),fused_ms=bench(lambda:fused(q,s,theta)))
 print(times,flush=True)
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
  for _ in range(5):fused(q,s,theta)
  torch.cuda.synchronize()
 prof.export_chrome_trace(str(out/'route_after.trace.json'))
 events=[dict(name=e.key,calls=e.count,self_device_us=getattr(e,'self_device_time_total',0)) for e in prof.key_averages() if getattr(e,'self_device_time_total',0)]
 (out/'route_after.profile.json').write_text(json.dumps(events,indent=2))
 (out/'fused_route_checks.json').write_text(json.dumps(dict(passed=True,rows=rows,timing=times),indent=2))
