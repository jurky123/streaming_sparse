import os,time,json,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4)
from wan.modules.model import rope_params
from wan.modules.causal_model import causal_rope_apply
from local_experiments.review_v1.fused_rope import apply
freqs=torch.cat([rope_params(1024,44),rope_params(1024,42),rope_params(1024,42)],1).cuda();grid=torch.tensor([[3,32,48]])
x=torch.load('/tmp/latest_kernel_optimization2/attention_inputs.pt',map_location='cuda',weights_only=True);q=x['q'].transpose(0,1).contiguous()[None];k=x['k'][:,-72:].flatten(1,2).transpose(0,1).contiguous()[None];rows=[]
for start in range(0,123,3):
 ref=(causal_rope_apply(q,grid,freqs,start),causal_rope_apply(k,grid,freqs,start));actual=apply(q,k,grid,freqs,start)
 row=dict(start_frame=start,bitwise_equal=[torch.equal(a,b) for a,b in zip(ref,actual)],max_abs=[float((a.float()-b.float()).abs().max()) for a,b in zip(ref,actual)]);rows.append(row);print(row,flush=True);assert all(row['bitwise_equal'])
def bench(fn):
 for _ in range(5):fn()
 torch.cuda.synchronize();start=time.perf_counter()
 for _ in range(100):fn()
 torch.cuda.synchronize();return (time.perf_counter()-start)*10
result=dict(passed=True,rows=rows,previous_ms=bench(lambda:(causal_rope_apply(q,grid,freqs,24),causal_rope_apply(k,grid,freqs,24))),fused_ms=bench(lambda:apply(q,k,grid,freqs,24)))
print(result,flush=True);Path('/tmp/latest_kernel_optimization3/rope_checks.json').write_text(json.dumps(result,indent=2))
