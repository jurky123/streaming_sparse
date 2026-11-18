"""Profile actual block pointwise operations and capture full-size operands."""
import json
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights,make_runtime
from .weights import ROOT
from .block_pointwise import PointwiseBlocks
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 cfg=config('base',block_pointwise=False,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg)
 m,r,p,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,m,r,p,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 z=torch.load(ROOT/'artifacts/video_runtime_v1/block_retained_v1/video/base_old/latents.pt',weights_only=True).cuda()
 for c in range(8):engine.forward(z[:,c*3:c*3+3],emb,c,4)
 captured={}
 def observe(layer,name,args,out):
  if layer==5 and engine.current['step']==0:captured[name]=[v.detach().clone() if isinstance(v,torch.Tensor) else v for v in args]
 pb=PointwiseBlocks(engine,observe=observe,profile=True)
 generate_chunk(engine,emb,701,8);pb.observe=None
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  # Same full window; clean commit is excluded from pointwise attribution below.
  engine.forward(z[:,24:27],emb,8,0);torch.cuda.synchronize()
 out=ROOT/'artifacts/video_runtime_v1/block_pointwise_v1';out.mkdir(exist_ok=True)
 torch.save({k:[v.cpu() if isinstance(v,torch.Tensor) else v for v in vs] for k,vs in captured.items()},out/'input.pt')
 prof.export_chrome_trace(str(out/'detail.trace.json'))
 rows=[dict(name=e.key,calls=e.count,device_us=e.device_time_total,self_device_us=e.self_device_time_total) for e in prof.key_averages() if e.key.startswith('pointwise.')]
 (out/'profile.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
 print({k:[str(v.dtype) if isinstance(v,torch.Tensor) else v for v in vs] for k,vs in captured.items()},flush=True)
 pb.close();rt.close()
if __name__=='__main__':main()
