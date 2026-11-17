"""Paired deployed route trials for review-proposed representation/denominator changes."""
import argparse,json
import torch
from .weights import ROOT
from .profile_route import inputs,capture
from .pipeline_graph import BatchGraphs
from .risk_projected import risk
from .deployment import current,configure
from . import pipeline_graph,reader_reduce
from ..review_v1.ops import Config,Summary
from .benchmark import measure
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile-json',action='store_true',default=True);ap.add_argument('--steady-profile',action='store_true',default=True);ap.add_argument('--variant',default='scan',choices=['scan']);ap.add_argument('--pages',type=int,default=432);a=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 configure(current());pipeline_graph.CACHED_LOGMASS=True
 ref,_,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
 saved=torch.load(ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True)
 e={k:v.cuda() for k,v in saved.items() if k not in ('summary','en')};p=a.pages
 e['s']=Summary(*(saved['summary'][n][:,:p].cuda().contiguous() for n in ('keys','mean','detail','stats')))
 e['en']={k:v[:,:p].cuda().contiguous() for k,v in saved['en'].items()};e['own']=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1).contiguous()
 root=ROOT/f'artifacts/video_runtime_v1/scan_fp32_v1/{a.variant}_p{p}{"_steady" if a.steady_profile else ""}{"_detail" if a.profile_json else ""}';root.mkdir(parents=True,exist_ok=True)
 rows={};graphs={};outs={};functions={}
 for name,enabled in [('before',False),('after',True)]:
  reader_reduce.OUTPUT32=True;reader_reduce.ENDPOINT_DEN=True;reader_reduce.WORK_ROWS=0;reader_reduce.SCAN_FP32=enabled
  from . import endpoint_den
  endpoint_den.CACHED_GROUPS=True
  ep=dict(e);ep['en']={k:(v.bfloat16() if k in ('co','A') else v) for k,v in e['en'].items()}
  rt=BatchGraphs(risk,ref.phi,Config(),compile_route=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True)
  for pages in [216,288,360]:
   if pages>=p:continue
   small=dict(ep);small['s']=Summary(*(getattr(ep['s'],n)[:,:pages].contiguous() for n in ('keys','mean','detail','stats')));small['en']={k:v[:,:pages].contiguous() for k,v in ep['en'].items()};small['own']=(torch.arange(72,device='cuda')+pages-72)[None].expand(12,-1).contiguous()
   g,o=capture(lambda:rt.compute(small,True))
  fn=lambda rt=rt,ep=ep:rt.compute(ep,True);functions[name]=fn;graphs[name],out=capture(fn);outs[name]=tuple(x.clone() for x in out)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:fn();torch.cuda.synchronize()
  (root/f'{name}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
 for _ in range(30):
  for g in graphs.values():g.replay()
 for n,g in graphs.items():rows[n]=measure(g.replay,50)
 if a.steady_profile:
  for n,f in functions.items():
   reader_reduce.OUTPUT32=True;reader_reduce.ENDPOINT_DEN=True;reader_reduce.SCAN_FP32=n=='after'
   endpoint_den.CACHED_GROUPS=True
   for _ in range(30):graphs[n].replay()
   torch.cuda.synchronize()
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:f();torch.cuda.synchronize()
   (root/f'{n}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=45))
   if a.profile_json:(root/f'{n}_profile.json').write_text(json.dumps([dict(name=x.key,self_cuda_us=x.self_device_time_total,count=x.count) for x in prof.key_averages()],indent=2))
 result=dict(rows=rows,speedup=rows['before']['gpu_ms_median']/rows['after']['gpu_ms_median'],output_equal=[torch.equal(x,y) for x,y in zip(outs['before'],outs['after'])],output_max_abs=[float((x.double()-y.double()).abs().max()) for x,y in zip(outs['before'],outs['after'])],variant=a.variant,pages=p,weights=identity)
 (root/'result.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='rows'},flush=True)
if __name__=='__main__':main()
