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
    times = []
    for rep in range(6):
        rt.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        z = generate(emb,701,7)
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        assert z.isfinite().all()
        print(method,times[-1],flush=True)
    Path(__file__).with_name('current_'+method+'.json').write_text(json.dumps(times))
run(sys.argv[1])
