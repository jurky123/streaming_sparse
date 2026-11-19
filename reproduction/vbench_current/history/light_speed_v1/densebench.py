import sys,time,json,torch
import os
from pathlib import Path
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.backbone_ops import install
root=Path(__file__).resolve().parent;method,scene=sys.argv[1],int(sys.argv[2])
@torch.inference_mode()
def run():
 torch.set_num_threads(2)
 cfg=config('base' if method in ('native','native_short') else method,local_assets='/tmp/streaming_sparse-0/runtime')
 if method in ('native','native_short'):cfg.mode='dense_native'
 configure(cfg);engine=Engine();engine.wrapper.requires_grad_(False)
 if method in ('native','native_short'):install(engine,cfg);rt=engine
 else:rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
 emb=torch.load(root.parents[1]/'embeddings'/f'{"short" if method=="native_short" else "long"}_{scene}.pt' ,weights_only=True).cuda()
 times=[]
 for rep in range(4):
  rt.reset();torch.cuda.synchronize();t=time.monotonic()
  z=torch.cat([generate_chunk(engine,emb,701+scene,c) for c in range(7 if method=='native_short' else 41)],1)
  torch.cuda.synchronize();times.append(time.monotonic()-t);assert z.isfinite().all()
  (root/f'{method}_{scene}{os.getenv("REF_SUFFIX","")}.json' ).write_text(json.dumps(dict(times=times)))
run()
