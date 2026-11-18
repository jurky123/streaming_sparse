"""Same real P432 input: base, three-head correction and full-head correction."""
import argparse,json,statistics
import torch
from .weights import ROOT,load
from .deployment import current,configure
from .pipeline_graph import BatchGraphs
from .risk_projected import risk
from .page import encode
from . import tensor_bmm
from .profile_route import capture
from ..review_v1.ops import Summary,Config,haar
@torch.no_grad()
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--native-route-qk", action="store_true");parser.add_argument("--scale-in-qk", action="store_true");parser.add_argument("--inline-u",action="store_true");args=parser.parse_args()
 if args.scale_in_qk:args.native_route_qk=True
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch._dynamo.config.cache_size_limit=64
 cfg=current();configure(cfg);model,reader,phi,B,identity=load(checkpoint=cfg.checkpoint,reader_checkpoint=cfg.reader_checkpoint)
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
  for flag in (False,True):
   tensor_bmm.QK_TILE64=True if args.native_route_qk else flag
   rt.native_route_qk=args.native_route_qk and flag
   rt.scale_in_qk=args.scale_in_qk and flag
   if args.inline_u:
    from .risk_projected import risk_inline
    rt.risk_function=risk_inline if flag else risk
    rt.native_route_qk=rt.scale_in_qk=tensor_bmm.QK_TILE64=True
   e['q']=q.bfloat16() if rt.native_route_qk else q
   g,out=capture(lambda:rt.compute(e,mode!='base'));graphs.append(g);values.append(tuple(v.clone() for v in out))
  for g in graphs:
   for _ in range(30):g.replay()
  torch.cuda.synchronize();times=[[],[]]
  for repeat in range(30):
   for i in ([0,1] if repeat%2==0 else [1,0]):
    start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True);start.record()
    for _ in range(10):graphs[i].replay()
    stop.record();stop.synchronize();times[i].append(start.elapsed_time(stop)/10)
  old,new=map(statistics.median,times);row=dict(mode=mode,before_ms=old,after_ms=new,speedup=old/new,bitwise=[torch.equal(a,b) for a,b in zip(*values)],samples_ms=times)
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
 out=ROOT/'artifacts/video_runtime_v1'/('risk_inline_u_v1' if args.inline_u else 'scaled_route_qk_v1' if args.scale_in_qk else 'native_route_qk_v1' if args.native_route_qk else 'qk_tiles_v1');out.mkdir(exist_ok=True);(out/'route.json').write_text(json.dumps(dict(rows=rows,identity=identity),indent=2))
if __name__=='__main__':main()
