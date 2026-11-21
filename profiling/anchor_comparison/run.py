import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
from phase import set_phase
from queue_policy import attach
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from local_experiments.video_runtime_v1.backbone_ops import install
@torch.no_grad()
def main(method):
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    native=method in ('dense','lightforcing','pisa','svgear','solattn')
    cfg=config('base' if native else method,local_assets=repo)
    if native:cfg.mode='dense_native'
    configure(cfg)
    engine=Engine()
    if native:
        install(engine,cfg)
        rt=SimpleNamespace(engine=engine,stores={},reset=engine.reset)
    else:
        rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    for b in engine.wrapper.model.blocks:b.self_attn.sink_size=1
    emb=torch.load(repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt',weights_only=True).cuda()
    def gen(embedding,seed,chunks):
        parts=[]
        for c in range(chunks):
            if c==6:canonical=torch.cat([x['k'][:,:1536] for x in engine.cache])
            if c>=6:set_phase(rt,canonical,3*c-15)
            parts.append(generate_chunk(engine,embedding,seed,c))
        return torch.cat(parts,1)
    if method.startswith('intersection'):gen=attach(rt,engine,10)
    from external import install_method
    hook=install_method(method,engine) if native and method!='dense' else None
    rows={}
    for n in [7,41]:
        secs=[]
        for _ in range(3):
            rt.reset()
            if hook:hook[0](n)
            torch.cuda.synchronize();start=time.perf_counter()
            z=gen(emb,701,n)
            torch.cuda.synchronize();secs.append(time.perf_counter()-start)
            assert z.isfinite().all()
            if hook:assert hook[1][0]==(n*150 if method=='lightforcing' else (n-2)*116)
            print(method,n,secs[-1],flush=True)
        rows[n]=secs
        Path(__file__).with_name((sys.argv[2] if len(sys.argv)>2 else method)+'.json').write_text(json.dumps(rows))
main(sys.argv[1])
