"""Bitwise and timing gates for graph grouping, selection and KV reuse."""
import os,json,time,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
from local_experiments.review_v1.ops import Summary,RouteOps,Config
from local_experiments.review_v1.model import Layer
from local_experiments.review_v1.runtime import GroupingOps
from local_experiments.review_v1.cached_ops import GraphGrouping,GraphSelection
from local_experiments.review_v1.fused_route import FusedRiskRoute
out=Path('/tmp/latest_kernel_optimization3');out.mkdir(exist_ok=True)
def bench(fn,n=40):
 for _ in range(5):fn()
 torch.cuda.synchronize();start=time.perf_counter()
 for _ in range(n):fn()
 torch.cuda.synchronize();return (time.perf_counter()-start)*1000/n
with torch.no_grad():
 item=torch.load('/tmp/latest_kernel_optimization2/summary_inputs.pt',weights_only=True,map_location='cuda')['cheap'];m=Layer(item['state']);m.load_state_dict(item['state']);k=item['k'].flatten(1,2).transpose(0,1)[None];v=item['v'].flatten(1,2).transpose(0,1)[None]
 cfg=Config(history_pages=360);group=GraphGrouping(cfg);rows=[]
 for changed in [False,True]:
  if changed:m.pool_k.add_(.03);m.bank.mul_(1.01);m.group_gate.add_(.02);v=v*.99
  ref=GroupingOps.group(m,k,v,cfg);actual=group(m,k,v);equal=[torch.equal(a,b) for a,b in zip(actual,ref)]
  row=dict(changed=changed,bitwise_equal=equal,previous_ms=bench(lambda:GroupingOps.group(m,k,v,cfg)),graph_ms=bench(lambda:group(m,k,v)));print('group',row,flush=True);rows.append(row);assert all(equal)
 (out/'grouping_checks.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
 x=torch.load('/tmp/latest_kernel_optimization2/attention_inputs.pt',weights_only=True,map_location='cuda');q=x['q'].reshape(12,72,64,128).index_select(2,torch.linspace(0,63,16,device='cuda').round().long()).flatten(1,2);s=Summary(**x['summary']);route=FusedRiskRoute();selector=GraphSelection(cfg);rows=[]
 for p in [216,288,360,432]:
  score=route(q,s.slice(0,p),m.theta);own=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1)
  for tied in [False,True]:
   if tied:score.zero_()
   ref=RouteOps.select(score,own,cfg);actual=selector(score);equal={n:torch.equal(ref[n],actual[n]) for n in ref}
   row=dict(pages=p,tied=tied,equal=equal,previous_ms=bench(lambda:RouteOps.select(score,own,cfg)),graph_ms=bench(lambda:selector(score)));print('selection',row,flush=True);rows.append(row);assert all(equal.values())
 (out/'selection_checks.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
 cache={n:torch.randn(1,432*64,12,128,device='cuda',dtype=torch.bfloat16) for n in ['k','v']};rows=[]
 for p in [0,72,144,216,288,360]:
  def cat():return torch.cat((cache['k'][:,:p*64],k),1),torch.cat((cache['v'][:,:p*64],v),1)
  def reuse():
   cache['k'][:,p*64:p*64+4608].copy_(k);cache['v'][:,p*64:p*64+4608].copy_(v)
   return cache['k'][:,:p*64+4608],cache['v'][:,:p*64+4608]
  ref=cat();actual=reuse();assert all(torch.equal(a,b) for a,b in zip(ref,actual))
  row=dict(history_pages=p,bitwise_equal=True,cat_ms=bench(cat),reuse_ms=bench(reuse));print('kv',row,flush=True);rows.append(row)
 (out/'kv_checks.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
