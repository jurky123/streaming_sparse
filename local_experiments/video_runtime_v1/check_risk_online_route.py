"""Risk online probability consumption: complete route contract and timing."""
import argparse,json,statistics
import torch
from .weights import ROOT,load
from .deployment import current,configure
from .pipeline_graph import BatchGraphs
from .risk_projected import risk
from .risk_online import risk as online
from .page import encode
from . import tensor_bmm
from .profile_route import capture
from ..review_v1.ops import Summary,Config,haar
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--fixed-pages',action='store_true')
 ap.add_argument('--matched',action='store_true')
 ap.add_argument('--page-route',action='store_true')
 args=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch._dynamo.config.cache_size_limit=64
 cfg=current();configure(cfg)
 if args.matched or args.fixed_pages or args.page_route:
  from . import pipeline_graph
  pipeline_graph.CACHED_LOGMASS=cfg.cached_logmass
 model,reader,phi,B,identity=load(checkpoint=cfg.checkpoint,reader_checkpoint=cfg.reader_checkpoint)
 data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
 offsets=torch.linspace(0,63,16,device='cuda').round().long()
 q=data['q'].cuda()[0].transpose(0,1).reshape(12,72,64,128).index_select(2,offsets).float()
 s=Summary(*(data['summary'][n].cuda() for n in ('keys','mean','detail','stats')));basis=model[15].basis
 co,A,mean=encode(s,reader,phi,B.double(),coeff_bf16=True);detail=torch.matmul(s.detail.double(),(basis.bfloat16().double().transpose(1,2)@B.double())[:,None])
 base=dict(q=q,s=s,en=dict(co=co,A=A,mean=mean,detail=detail),basis=basis,theta=model[15].theta,own=(torch.arange(72,device='cuda')+360)[None].expand(12,-1),offsets=offsets,haar=haar(q).double(),haar_float=haar(q))
 rows=[]
 for mode in ('base','intersection','full'):
  e=dict(base)
  if mode=='intersection':e['active_heads']=torch.tensor([4,3,9],device='cuda')
  rt=BatchGraphs(risk,phi,Config(),compile_route=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True,static_route=True,input_m16=True)
  graphs=[];values=[]
  if args.fixed_pages or args.matched or args.page_route:e['q']=q.bfloat16()
  for flag in (False,True):
   if args.fixed_pages or args.matched or args.page_route:
    from .risk_online import matched_risk
    from .risk_projected import risk_inline
    rt.risk_function=matched_risk if args.fixed_pages or (flag and args.matched) else risk_inline
    rt.matched_probabilities=flag and args.page_route
    rt.native_route_qk=rt.scale_in_qk=True
    rt.reuse_selected_ids='ties'
    rt.fixed_page_sets=flag and (args.fixed_pages or args.page_route)
   else:rt.risk_function=online if flag else risk
   g,out=capture(lambda:rt.compute(e,mode!='base'));graphs.append(g);values.append(tuple(v.clone() for v in out))
  for g in graphs:
   for _ in range(30):g.replay()
  torch.cuda.synchronize();times=[[],[]]
  for repeat in range(30):
   for i in ([0,1] if repeat%2==0 else [1,0]):
    start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True);start.record()
    for _ in range(10):graphs[i].replay()
    stop.record();stop.synchronize();times[i].append(start.elapsed_time(stop)/10)
  old,new=map(statistics.median,times);row=dict(mode=mode,before_ms=old,after_ms=new,speedup=old/new,bitwise=[torch.equal(a,b) for a,b in zip(*values)],risk_max_abs=float((values[0][3]-values[1][3]).abs().max()),samples_ms=times)
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
 out=ROOT/'artifacts/video_runtime_v1'/('page_route_v1' if args.page_route else 'fixed_page_select_v1' if args.fixed_pages else 'risk_matched_norm_v1' if args.matched else 'post_rope_round2');out.mkdir(exist_ok=True);(out/('route_cached.json' if args.matched or args.fixed_pages or args.page_route else 'route.json')).write_text(json.dumps(dict(rows=rows,identity=identity),indent=2))
if __name__=='__main__':main()
