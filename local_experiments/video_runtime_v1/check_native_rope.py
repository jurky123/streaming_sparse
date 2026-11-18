"""Native RoPE equality and full-size isolated timing on actual teacher frequencies."""
import json,statistics
import torch
from .weights import ROOT
from .accepted_tiers import config
from .deployment import configure
from ..broad_common import Engine
from .native_rope import apply
from .profile_route import capture
@torch.no_grad()
def main():
 torch.set_num_threads(4);configure(config('base',local_assets='/tmp/streaming_sparse-0/runtime'))
 engine=Engine();engine.wrapper.requires_grad_(False)
 from wan.modules.causal_model import causal_rope_apply
 freq=engine.wrapper.model.freqs.cuda();grid=torch.tensor([[3,32,48]])
 data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True);q=data['q'].cuda();rows=[]
 for start in range(0,27,3):
  old=causal_rope_apply(q,grid,freq,start_frame=start);new=apply(q,grid,freq,start)
  rows.append(dict(start=start,bitwise=torch.equal(old,new),max_abs=(old-new).abs().max().item()))
 graphs=[capture(lambda:causal_rope_apply(q,grid,freq,start_frame=24))[0],capture(lambda:apply(q,grid,freq,24))[0]];times=[[],[]]
 for g in graphs:
  for _ in range(50):g.replay()
 torch.cuda.synchronize()
 for r in range(30):
  for i in ([0,1] if r%2==0 else [1,0]):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(10):graphs[i].replay()
   b.record();b.synchronize();times[i].append(a.elapsed_time(b)/10)
 old,new=map(statistics.median,times);out=ROOT/'artifacts/video_runtime_v1/fair_dense_v1';out.mkdir(exist_ok=True)
 result=dict(contracts=rows,passed=all(r['bitwise'] for r in rows),before_ms=old,after_ms=new,speedup=old/new,samples_ms=times);(out/'rope_module.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='samples_ms'})
if __name__=='__main__':main()
