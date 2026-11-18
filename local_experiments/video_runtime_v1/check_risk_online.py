"""Risk probability materialization versus row-normalizer consumption."""
import argparse,json,statistics
import torch
from .weights import ROOT,load
from .deployment import current,configure
from .profile_route import capture
from .tensor_bmm import bf16_inputs_bmm
from .risk_projected import risk as old
from .risk_online import risk as new
from ..review_v1.ops import Summary,RouteOps,Config
@torch.no_grad()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--inline-u',action='store_true');parser.add_argument('--probe-row',action='store_true');parser.add_argument('--matched',action='store_true');parser.add_argument('--direct-bound',action='store_true');args=parser.parse_args()
 candidate=new
 if args.inline_u:
  from .risk_projected import risk_inline as candidate
 if args.probe_row:
  from .risk_probe_row import risk as candidate
  from .risk_projected import risk_inline as reference
 elif args.matched:
  from .risk_online import matched_risk as candidate
  from .risk_projected import risk_inline as reference
 else:reference=old
 if args.direct_bound:
  from .risk_online import matched_risk as reference, bounded_risk as candidate
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch._dynamo.config.cache_size_limit=64
 cfg=current();configure(cfg);model,*_=load(checkpoint=cfg.checkpoint,reader_checkpoint=cfg.reader_checkpoint)
 data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True);q=data['q'].cuda()[0].transpose(0,1).reshape(12,72,64,128).index_select(2,torch.linspace(0,63,16,device='cuda').round().long()).float()
 root=ROOT/'artifacts/video_runtime_v1'/('risk_bound_v1' if args.direct_bound else 'risk_matched_norm_v1' if args.matched else 'risk_probe_row_v1' if args.probe_row else 'risk_inline_u_v1' if args.inline_u else 'post_rope_round2');root.mkdir(exist_ok=True);rows=[]
 for pages in (216,288,360,432):
  s=Summary(*(data['summary'][n][:,:pages].cuda().contiguous() for n in ('keys','mean','detail','stats')));z=bf16_inputs_bmm(q.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(12,72,16,pages,4)/(128**.5);theta=model[15].theta
  graphs=[];values=[];functions=[]
  for fn in (reference,candidate):
   f=torch.compile(lambda fn=fn:fn(q,z,s,theta),fullgraph=True,options={'triton.cudagraphs':False});functions.append(f);g,out=capture(f);graphs.append(g);values.append(out.clone())
  for g in graphs:
   for _ in range(30):g.replay()
  times=[[],[]]
  for rep in range(30):
   for i in ([0,1] if rep%2==0 else [1,0]):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
    for _ in range(20):graphs[i].replay()
    b.record();b.synchronize();times[i].append(a.elapsed_time(b)/20)
  x,y=values;before,after=map(statistics.median,times);own=(torch.arange(72,device='cuda')+pages-72)[None].expand(12,-1)
  mask=[RouteOps.select(v,own,Config(),check_valid=False)['mask'] for v in values]
  row=dict(pages=pages,before_ms=before,after_ms=after,speedup=before/after,bitwise=torch.equal(x,y),max_abs=(x-y).abs().max().item(),nmse=((x-y).square().sum()/x.square().sum()).item(),mask_equal=torch.equal(*mask),samples_ms=times);rows.append(row);print({k:v for k,v in row.items() if k!='samples_ms'},flush=True)
  if pages==432:
   for label,f in zip(['old','new'],functions):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
     for _ in range(20):f()
     torch.cuda.synchronize()
    if args.direct_bound:assert any(e.key=='_bound' for e in prof.key_averages())==(label=='new')
    prof.export_chrome_trace(str(root/(label+'.trace.json')));(root/(label+'_profile.json')).write_text(json.dumps([dict(name=e.key,calls=e.count,self_us=e.self_device_time_total) for e in prof.key_averages()],indent=2))
 (root/'module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
