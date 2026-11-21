import json
import sys
import time
from pathlib import Path
import torch

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / 'reproduction/vbench_current'))
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure, make_runtime, load_weights
from local_experiments.video_runtime_v1.backbone_ops import install
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from queue_policy import attach

@torch.no_grad()
def run(method):
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = config('base' if method == 'dense' else method, local_assets=repo)
    if method == 'dense':
        cfg.mode = 'dense_native'
    configure(cfg)
    engine = Engine()
    if method == 'dense':
        install(engine, cfg)
        rt = engine
    else:
        rt = make_runtime(engine, *load_weights(cfg)[:4], cfg)
    generate = attach(rt, engine, 10) if method in ('intersection','intersection_fast') else lambda e,s,n: torch.cat([generate_chunk(engine,e,s,i) for i in range(n)],1)
    emb = torch.load(repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt',weights_only=True).cuda()
    from local_experiments.stream_eval_common import native_vae,decode
    import numpy as np
    rt.reset()
    z=generate(emb,701,7)
    assert z.isfinite().all()
    frames=np.concatenate(decode(native_vae(),z,80))
    assert frames.shape==(80,512,768,3)
    np.save('/tmp/compare_short721_'+method+'.npy',frames)
run(sys.argv[1])
