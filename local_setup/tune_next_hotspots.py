"""Real-input tuning with numeric gates; diagnostics only, no end-to-end claim."""
import os,json,time,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
from local_experiments.review_v1.ops import Summary,SummaryOps
from local_experiments.review_v1.model import Layer
from local_experiments.review_v1.fast_attention import attention as previous
from local_experiments.review_v1.tuned_attention import attention
from local_experiments.review_v1.graph_summary import GraphSummary
from local_experiments.review_v1.fused_route import FusedRiskRoute
root=Path('/tmp/latest_kernel_optimization2')
def bench(fn,n=30):
 for _ in range(3):fn()
 torch.cuda.synchronize();start=time.perf_counter()
 for _ in range(n):fn()
 torch.cuda.synchronize();return (time.perf_counter()-start)*1000/n
with torch.no_grad():
 sample=torch.load(root/'attention_inputs.pt',map_location='cuda',weights_only=True);q=sample['q'];k=sample['k'];v=sample['v'];s=Summary(**sample['summary']);basis=sample['basis'];sel=sample['selection'];args=(q,k,v,s,basis,sel)
 ref=previous(*args);rows=[];old_ms=bench(lambda:previous(*args));print('old_attention_ms',old_ms,flush=True)
 for bf16 in ([] if os.environ.get("SKIP_ATTENTION") else [False,True]):
  for bq,br in [(64,64),(32,64),(64,128),(32,128),(64,256),(32,256)]:
   try:
    out=attention(*args,bq=bq,br=br,bf16_av=bf16);diff=out.float()-ref.float();nmse=float(diff.square().mean()/ref.float().square().mean());err=float(diff.abs().max());passed=bool(out.isfinite().all()) and nmse<2e-5 and err<=.05
    row=dict(bq=bq,br=br,bf16_av=bf16,nmse=nmse,max_abs=err,bitwise_equal=torch.equal(out,ref),passed=passed,ms=bench(lambda:attention(*args,bq=bq,br=br,bf16_av=bf16)))
    rows.append(row);print(row,flush=True)
   except Exception as exc:
    row=dict(bq=bq,br=br,bf16_av=bf16,passed=False,error=str(exc));rows.append(row);print(row,flush=True)
 if rows:(root/'attention_tuning.json').write_text(json.dumps(dict(previous_ms=old_ms,candidates=rows),indent=2))
 graph=GraphSummary();rows=[];ss=torch.load(root/'summary_inputs.pt',map_location='cuda',weights_only=True)
 for mode,item in ss.items():
  k=item['k'];v=item['v'];m=Layer(item['state']);m.load_state_dict(item['state']);cheap=mode=='cheap'
  for change in [False,True]:
   if change:m.pool_k.add_(.03);v=v*.99
   ref=SummaryOps.build(m,k,v,cheap);out=graph(m,k,v,cheap)
   equal={n:torch.equal(getattr(ref,n),getattr(out,n)) for n in ['keys','mean','detail','stats']}
   errors={n:float((getattr(ref,n).float()-getattr(out,n).float()).square().mean()/getattr(ref,n).float().square().mean().clamp_min(1e-12)) for n in equal}
   row=dict(mode=mode,changed_inputs_and_weights=change,equal=equal,nmse=errors,previous_ms=bench(lambda:SummaryOps.build(m,k,v,cheap)),graph_ms=bench(lambda:graph(m,k,v,cheap)))
   print(row,flush=True);rows.append(row);assert all(equal.values()),row
 (root/'summary_graph_checks.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
 # Batch all 72 tiles without changing the independent query reductions.
 route=FusedRiskRoute();theta=Layer(ss['cheap']['state']).theta
 probes=q.reshape(12,72,64,128).index_select(2,torch.linspace(0,63,16,device='cuda').round().long()).flatten(1,2)
 def oldroute():return torch.cat([route(probes[:,i:i+128],s,theta) for i in range(0,1152,128)],1)
 ref=oldroute();out=route(probes,s,theta);assert torch.equal(ref,out)
 row=dict(bitwise_equal=True,previous_ms=bench(oldroute),batched_ms=bench(lambda:route(probes,s,theta)))
 print(row,flush=True);(root/'batched_route_checks.json').write_text(json.dumps(row,indent=2))
