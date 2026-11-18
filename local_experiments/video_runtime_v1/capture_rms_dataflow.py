"""Capture real Q/K RMSNorm operands and profile the unchanged mean-based path."""
import json
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights,make_runtime
from .weights import ROOT
from .rms_dataflow import RMSDataflow
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 cfg=config('base',rms_dataflow=False,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg)
 m,r,p,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,m,r,p,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 z=torch.load(ROOT/'artifacts/video_runtime_v1/block_pointwise_v1/video/base_new/latents.pt',weights_only=True).cuda()
 captured={}
 def observe(name,x,w,eps):
  if name.startswith('blocks.5.') and (engine.current['step']==0 or 'cross_attn.norm_k' in name) and name not in captured:
   captured[name]=dict(x=x.detach().clone(),weight=w.detach().clone(),eps=eps)
 patch=RMSDataflow(engine,optimized=False,observe=observe)
 for c in range(8):engine.forward(z[:,c*3:c*3+3],emb,c,4)
 generate_chunk(engine,emb,701,8);patch.observe=None;patch.profile=True
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  engine.forward(z[:,24:27],emb,8,0);torch.cuda.synchronize()
 out=ROOT/'artifacts/video_runtime_v1/rms_dataflow_v1';out.mkdir(exist_ok=True)
 torch.save({k:{n:v.cpu() if isinstance(v,torch.Tensor) else v for n,v in d.items()} for k,d in captured.items()},out/'input.pt')
 prof.export_chrome_trace(str(out/'detail.trace.json'))
 rows=[dict(name=e.key,calls=e.count,device_us=e.device_time_total,self_device_us=e.self_device_time_total) for e in prof.key_averages() if e.key.startswith('rms.')]
 (out/'profile.json').write_text(json.dumps(rows,indent=2));print({k:(tuple(d['x'].shape),str(d['x'].dtype),str(d['weight'].dtype)) for k,d in captured.items()},flush=True)
 patch.close();rt.close()
if __name__=='__main__':main()
