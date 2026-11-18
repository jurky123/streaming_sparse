"""Bounded pool/statistics dataflow experiment, including full summary timing."""
import argparse,json,statistics
from functools import partial
from pathlib import Path
import torch
from .weights import ROOT,load
from .profile_route import capture
from ..review_v1.split_summary import build as old
from .summary_moments import build as new
@torch.no_grad()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--aligned-aux',action='store_true');parser.add_argument('--project-features',action='store_true');parser.add_argument('--fuse-stats',action='store_true');parser.add_argument('--partial-page',action='store_true');parser.add_argument('--feature-page',action='store_true');parser.add_argument('--pool-warps',type=int,choices=(1,2,8));parser.add_argument('--match-pool',action='store_true');parser.add_argument('--cuda-pool',action='store_true');args=parser.parse_args()
 variants=(new,partial(new,aligned_aux=True)) if args.aligned_aux else (old,new)
 if args.cuda_pool:variants=(new,partial(new,cuda_pool=True))
 if args.pool_warps:variants=(new,partial(new,pool_warps=args.pool_warps,match_pool=args.match_pool))
 if args.feature_page:variants=(new,partial(new,feature_page=True))
 if args.partial_page:variants=(new,partial(new,partial_page=True))
 if args.fuse_stats:variants=(new,partial(new,fuse_stats=True))
 if args.project_features:variants=(new,partial(new,project_features=True))
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 model,*_=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt')
 path=next((ROOT/'artifacts/full_coverage_v1/data/city_walk_s701').glob('c8_l15_s0_h0_t*.pt'));ev=torch.load(path,weights_only=True);kv=torch.load(path.parent/ev['kv_file'],weights_only=True)
 k=kv['k'][-72:].cuda().bfloat16()[None].expand(12,-1,-1,-1).contiguous();v=kv['v'][-72:].cuda().bfloat16()[None].expand_as(k).contiguous()
 root=ROOT/'artifacts/video_runtime_v1'/('cuda_pool_v1' if args.cuda_pool else f'pool_{"matched_" if args.match_pool else ""}warps_{args.pool_warps}_v1' if args.pool_warps else 'feature_page_v1' if args.feature_page else 'partial_page_v1' if args.partial_page else 'page_fused_stats_v1' if args.fuse_stats else 'pool_feature_scalar_v1' if args.project_features else 'aligned_aux_v1' if args.aligned_aux else 'post_rope_round1');root.mkdir(exist_ok=True);rows=[]
 for native in (False,True):
  if native:
   k=k.flatten(1,2).transpose(0,1).contiguous().transpose(0,1).reshape(12,72,64,128);v=v.flatten(1,2).transpose(0,1).contiguous().transpose(0,1).reshape(12,72,64,128)
  for cheap in (True,False):
   graphs=[];outputs=[];times=[[],[]]
   for fn in variants:
    g,o=capture(lambda:fn(model[15],k,v,cheap));graphs.append(g);outputs.append(o)
   for g in graphs:
    for _ in range(30):g.replay()
   for rep in range(30):
    for i in ([0,1] if rep%2==0 else [1,0]):
     a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
     for _ in range(20):graphs[i].replay()
     b.record();b.synchronize();times[i].append(a.elapsed_time(b)/20)
   before,after=map(statistics.median,times)
   checks={n:torch.equal(getattr(outputs[0],n),getattr(outputs[1],n)) for n in ['keys','mean','detail','stats']}
   rows.append(dict(native=native,cheap=cheap,before_ms=before,after_ms=after,speedup=before/after,checks=checks,samples_ms=times));print({k:v for k,v in rows[-1].items() if k!='samples_ms'},flush=True)
   if native and cheap:
    for label,fn in zip(('old','new'),variants):
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
      for _ in range(20):fn(model[15],k,v,cheap)
      torch.cuda.synchronize()
     p.export_chrome_trace(str(root/(label+'.trace.json')))
     (root/(label+'_profile.json')).write_text(json.dumps([dict(name=e.key,calls=e.count,self_us=e.self_device_time_total) for e in p.key_averages()],indent=2))
 (root/'module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
