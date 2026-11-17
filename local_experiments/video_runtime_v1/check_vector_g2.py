"""Bounded vector-only trial: G1 vs G2, unchanged full WORK and scalar."""
import json,argparse
import torch
from .weights import ROOT
from .deployment import current,configure
from .profile_route import inputs,capture
from .pipeline_graph import BatchGraphs
from .risk_projected import risk
from .logmass_cached import normalized
from .tensor_bmm import bf16_inputs_bmm
from .scores import candidate_pool
from . import pipeline_graph,reader_reduce as rr
from ..review_v1.ops import Config,Summary,RouteOps
from .benchmark import measure
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='artifacts/video_runtime_v1/vector_g2_v1');cli=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;configure(current());pipeline_graph.CACHED_LOGMASS=True;rr.SCAN_FP32=True
 ref,_,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
 saved=torch.load(ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True)
 e={k:v.cuda() for k,v in saved.items() if k not in ('summary','en')};e['s']=Summary(*(saved['summary'][n].cuda() for n in ('keys','mean','detail','stats')));e['en']={k:(v.cuda().bfloat16() if k in ('co','A') else v.cuda()) for k,v in saved['en'].items()}
 q=e['q'].float().index_select(2,e['offsets']);h,t,m,d=q.shape;p=432;b=h*t;s=e['s'];en=e['en'];assert t==72
 z=bf16_inputs_bmm(q.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,m,p,4)/(d**.5);lm=normalized(z);u=ref.phi.query(q);rv=risk(q,z,s,e['theta']);mask=RouteOps.select(rv,e['own'],Config(),check_valid=False)['mask'];r,a=candidate_pool(rv,mask,e['own'])
 mapping=torch.full((b,p),-1,device='cuda',dtype=torch.int32);mapping=mapping.scatter(1,r.reshape(b,16),torch.arange(16,device='cuda',dtype=torch.int32)[None].expand(b,-1));mapping=mapping.scatter(1,a.reshape(b,32),torch.arange(16,48,device='cuda',dtype=torch.int32)[None].expand(b,-1))
 root=ROOT/cli.output;root.mkdir(parents=True,exist_ok=True);graphs={};out={};rows={};fns={};raw={}
 for size in [1,2]:
  rr.VECTOR_GROUP=size
  fn=lambda size=size:rr.reader_reduce(u,en['co'],en['A'],en['mean'],en['detail'],z,lm,mask.reshape(b,p),mapping,48,128,4,True,1,False,True,0,size)
  raw[size],values=capture(fn);out[('reader',size)]=tuple(v.clone() for v in values)
  rows[size]=dict(work_bytes=rr.KERNEL_STATS[(t,p)]['intermediate_bytes'],kernel_stats=dict(rr.KERNEL_STATS[(t,p)]))
  rt=BatchGraphs(risk,ref.phi,Config(),compile_route=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True)
  f=lambda rt=rt:rt.compute(e,True);graphs[size],values=capture(f);out[('route',size)]=tuple(v.clone() for v in values);fns[size]=f
 for _ in range(30):
  for size in [1,2]:graphs[size].replay();raw[size].replay()
 for size in [1,2]:
  rr.VECTOR_GROUP=size;rows[size]['route']=measure(graphs[size].replay,50);rows[size]['reader']=measure(raw[size].replay,50)
  rows[size]['reader_equal']=[torch.equal(x,y) for x,y in zip(out['reader',1],out['reader',size])];rows[size]['route_equal']=[torch.equal(x,y) for x,y in zip(out['route',1],out['route',size])]
  for _ in range(20):graphs[size].replay()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:fns[size]();torch.cuda.synchronize()
  (root/f'rows{size}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
  (root/f'rows{size}_profile.json').write_text(json.dumps([dict(name=x.key,cuda_us=x.self_device_time_total,count=x.count) for x in prof.key_averages()],indent=2))
  print(size,rows[size]['reader']['gpu_ms_median'],rows[size]['route']['gpu_ms_median'],rows[size]['reader_equal'],flush=True)
 stability=[]
 for order in [[2,1],[1,2],[2,1]]:
  trial={}
  for size in order:
   for _ in range(30):graphs[size].replay()
   trial[size]=dict(route=measure(graphs[size].replay,50),reader=measure(raw[size].replay,50))
  stability.append(trial)
 (root/'stability.json').write_text(json.dumps(stability,indent=2))
 (root/'result.json').write_text(json.dumps(dict(rows=rows,outer_tile_batch=72,pages=432,scope='one captured event; G1 and G2 only',weights=identity),indent=2))
if __name__=='__main__':main()
