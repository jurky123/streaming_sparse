"""Capture real clean-commit grouping inputs for isolated profiling."""
import json
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights,make_runtime
from .weights import ROOT
from ..broad_common import Engine
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg)
 model,reader,phi,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 z=torch.load(ROOT/'artifacts/video_runtime_v1/retained_tiers_profile_v1/base/latents.pt',weights_only=True).cuda()
 out=ROOT/'artifacts/video_runtime_v1/grouping_dataflow_v1';out.mkdir(exist_ok=True)
 original=rt.group_graph
 def capture(m,k,v):
  if engine.current['chunk']==8 and m is model[15]:
   torch.save(dict(k=k.cpu(),v=v.cpu(),layer=15,model={n:t.cpu() for n,t in m.state_dict().items()}),out/'input.pt')
  return original(m,k,v)
 rt.group_graph=capture
 for c in range(9):engine.forward(z[:,c*3:c*3+3],emb,c,4)
 rt.close();print(out/'input.pt')
if __name__=='__main__':main()
