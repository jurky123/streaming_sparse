"""Capture a real full-window attention event for isolated tail experiments."""
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights,make_runtime
from .weights import ROOT
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
from . import legacy_components
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg)
 model,reader,phi,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 z=torch.load(ROOT/'artifacts/video_runtime_v1/group_softmax_v1/video/base_new/latents.pt',weights_only=True).cuda()
 for c in range(8):engine.forward(z[:,c*3:c*3+3],emb,c,4)
 captured={};old=legacy_components.attention
 def capture(q,k,v,s,basis,ids,workspace,**kwargs):
  if basis is model[15].basis and engine.current['step']==0:
   captured.update({name:value.detach().cpu() for name,value in dict(q=q,k=k,v=v,basis=basis,ids=ids).items()});captured['summary']={name:getattr(s,name).detach().cpu() for name in ('keys','mean','detail','stats')}
  return old(q,k,v,s,basis,ids,workspace,**kwargs)
 legacy_components.attention=capture;generate_chunk(engine,emb,701,8)
 out=ROOT/'artifacts/video_runtime_v1/tail_liveness_v1';out.mkdir(exist_ok=True);torch.save(captured,out/'input.pt');rt.close();print(out/'input.pt')
if __name__=='__main__':main()
