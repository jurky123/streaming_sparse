"""Check that subset policy results do not depend on prior resident jobs."""
import argparse,json
import torch
from .weights import ROOT
from .deployment import current,configure,load_weights,make_runtime
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--head-first',action='store_true');p.add_argument('--static',action='store_true');a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch._dynamo.config.cache_size_limit=64;torch._dynamo.config.accumulated_cache_size_limit=512
 cfg=current(static_route=a.static,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg);model,reader,phi,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 def gen():
  rt.reset();return torch.cat([generate_chunk(engine,emb,701,c) for c in range(9)],1)
 if a.head_first:rt.set_pset_scope(heads=[0]);gen()
 rt.set_pset_scope(steps=[3]);gen();z=gen();r=ROOT/'artifacts/video_runtime_v1/pset_scope_v1/order_check';r.mkdir(exist_ok=True)
 name=f'static{int(a.static)}_headfirst{int(a.head_first)}';torch.save(z.cpu(),r/(name+'.pt'));(r/(name+'.json')).write_text(json.dumps(dict(counters=rt.counters.cpu().tolist(),static=a.static,head_first=a.head_first),indent=2));rt.close()
if __name__=='__main__':main()
