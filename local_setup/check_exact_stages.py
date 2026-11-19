import os,json,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4)
from local_experiments.review_v1.ops import Summary,RouteOps,Config
from local_experiments.review_v1.fast_attention import attention
from local_experiments.review_v1.tuned_attention import attention as tail
x=torch.load('/tmp/latest_kernel_optimization2/attention_inputs.pt',map_location='cuda',weights_only=True);q=x['q'];rows=[]
with torch.no_grad():
 for p in [216,288,360,432]:
  s=Summary(**{n:v[:,:p].contiguous() for n,v in x['summary'].items()});k=x['k'][:,:p];v=x['v'][:,:p]
  own=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1);scores=torch.rand(12,72,p,device='cuda');sel=RouteOps.select(scores,own,Config(history_pages=360));sel['fixed_count']=int(p*.1)
  for name,fn in [('stable',attention),('fast_tail',lambda *a,**kw:tail(*a,bf16_av=True,**kw))]:
   a=fn(q,k,v,s,x['basis'],sel,exact_stages=2);b=fn(q,k,v,s,x['basis'],sel,exact_stages=3)
   row=dict(pages=p,attention=name,bitwise_equal=torch.equal(a,b));assert row['bitwise_equal'];rows.append(row);print(row,flush=True)
Path('/tmp/latest_kernel_optimization3/exact_stage_checks.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
