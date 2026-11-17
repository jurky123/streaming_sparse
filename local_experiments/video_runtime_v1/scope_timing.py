"""Same-GPU paired steady timings for controls and all five selected policies."""
import argparse,json,time,statistics,os
import torch
from .weights import ROOT
from .scope_experiment import OUT,configure_scope,write
from .deployment import current,configure,load_weights,make_runtime
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();torch._dynamo.config.cache_size_limit=64;torch._dynamo.config.accumulated_cache_size_limit=512;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=current(static_route=True,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg);model,reader,phi,B,weights=load_weights(cfg)
 engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 specs=[dict(name='base'),dict(name='full')]+[j for j in json.loads((OUT/'selected_manifest.json').read_text()) if j['seed']==a.seed]
 def generate():
  rt.reset()
  for c in range(9):generate_chunk(engine,emb,a.seed,c)
  torch.cuda.synchronize()
 try:
  for spec in specs:
   path=OUT/'timing'/f's{a.seed}'/(spec['name']+'.json')
   if path.exists():continue
   configure_scope(rt,spec);generate();seconds=[]
   for _ in range(3):
    torch.cuda.synchronize();t=time.perf_counter();generate();seconds.append(time.perf_counter()-t)
   write(path,dict(policy=spec,seed=a.seed,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),seconds=seconds,median_seconds=statistics.median(seconds),counters=rt.counters.cpu().tolist(),scope='same GPU per seed, one full warmup then 3 repetitions, generation only excluding IO/VAE'))
   print(spec['name'],a.seed,seconds,flush=True)
 finally:rt.close()
if __name__=='__main__':main()
