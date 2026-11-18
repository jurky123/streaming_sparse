"""Count fallback CTAs on identical production inputs for three implementations."""
import ctypes
import json
from pathlib import Path

import torch

from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure, load_weights, make_runtime
from local_experiments.video_runtime_v1.fixed15_attention import PackedAttention, library
from local_experiments.video_runtime_v1.weights import ROOT

names = ('control', 'fused_center', 'floor_center')
original = PackedAttention.__call__
kernels = []
for name in names[1:]:
    kernel = ctypes.CDLL('/tmp/jzk_delivery_'+name+'.so').run
    kernel.argtypes = library().run.argtypes
    kernels.append(kernel)
rows = []

def audit(self, store, q, k, v, summary, basis, ids, workspace):
    result = original(self,store,q,k,v,summary,basis,ids,workspace)
    flags = workspace.tensor('packed_guard',(12,72),q.device,torch.int32)
    counts = [flags.sum()]
    pages = k.shape[1]//64
    packed = workspace.tensor('packed_summary',(12,(pages+15)//16,10624),q.device,q.dtype)
    scratch = torch.empty_like(result)
    alternate = torch.empty_like(flags)
    pk,pv = self.cache(store)
    basis = basis.bfloat16().contiguous()
    for kernel in kernels:
        args = (q,pk,pv,ids,scratch,packed,basis,alternate)
        assert kernel(*(x.data_ptr() for x in args),pages,ids.shape[-1],torch.cuda.current_stream().cuda_stream)==0
        counts.append(alternate.sum())
    rows.append(torch.stack(counts))
    return result

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    runtime=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    PackedAttention.__call__=audit
    try:
        for chunk in range(9):
            generate_chunk(engine,emb,701,chunk)
        counts=torch.stack(rows).cpu()
        assert counts.shape==(812,3)
        report={name:dict(fallback_tiles=int(c.sum()),calls_with_fallback=int((c>0).sum()),per_call=c.tolist()) for name,c in zip(names,counts.T)}
        (Path(__file__).parent/'trajectory_guard.json').write_text(json.dumps(report,indent=2))
    finally:
        PackedAttention.__call__=original
        runtime.close()

if __name__=='__main__':
    main()
