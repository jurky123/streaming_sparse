import json,os,sys,time
# The 480-frame comparison videos all use the same physical card.
if sys.argv[2] in ('before','after'):
    os.environ['CUDA_VISIBLE_DEVICES']='0'

from pathlib import Path
from types import SimpleNamespace
import torch
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
folder=Path(__file__).resolve().parent
sys.path.insert(0,str(repo/'profiling/anchor_comparison'))
from phase import set_phase
from queue_policy import attach
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from local_experiments.video_runtime_v1.backbone_ops import install
@torch.no_grad()
def main(method):
    variant=sys.argv[2]
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    native=method in ('dense','lightforcing','pisa','svgear','solattn')
    cfg=config('base' if native else method,local_assets=repo)
    if native:cfg.mode='dense_native'
    configure(cfg)
    torch._dynamo.config.cache_size_limit=512
    torch._dynamo.config.accumulated_cache_size_limit=4096
    engine=Engine()
    if native:
        install(engine,cfg)
        rt=SimpleNamespace(engine=engine,stores={},reset=engine.reset)
    else:
        rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    rt.sparse_clean = variant in ('after','scope')
    for b in engine.wrapper.model.blocks:b.self_attn.sink_size=1
    emb=torch.load(repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt',weights_only=True).cuda()
    def gen(embedding,seed,chunks):
        parts=[]
        for c in range(chunks):
            if c==6:canonical=torch.cat([x['k'][:,:1536] for x in engine.cache])
            if c>=6:set_phase(rt,canonical,3*c-15)
            parts.append(generate_chunk(engine,embedding,seed,c))
        return torch.cat(parts,1)
    if method.startswith('intersection'):
        if variant in ('after','scope'):
            from clean_policy import attach as attach_clean
            gen=attach_clean(rt,engine,10)
        else:
            gen=attach(rt,engine,10)
    from external import install_method
    hook=install_method(method,engine) if native and method!='dense' else None

    if method=='lightforcing' and variant in ('after','scope'):
        import wan.modules.causal_model as causal
        light_dispatch=causal.attention
        def matched(q,k,v,*args,**kwargs):
            if engine.current['chunk']<2 or engine.current['layer']==0:
                return engine.dense(q,k,v,*args,**kwargs)
            return light_dispatch(q,k,v,*args,**kwargs)
        causal.attention=matched
    from local_experiments.stream_eval_common import native_vae, decode, save_video
    import numpy as np
    if variant=='scope':
        from scope import search
        search(rt, gen, emb, method, folder, native_vae, decode)
        return
    if method=='intersection_fast' and variant in ('after','scope'):
        from scope import apply_selected
        apply_selected(rt,folder)
    if '--timing' in sys.argv:
        rows={}
        for chunks in (7,41):
            times=[]
            for _ in range(3):
                rt.reset()
                if hook:hook[0](chunks)
                torch.cuda.synchronize()
                start=time.perf_counter()
                z=gen(emb,701,chunks)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
                assert z.isfinite().all()
            rows[chunks]=times
            (folder/(method+'_'+variant+'_times.json')).write_text(json.dumps(rows))
        return
    chunks=7 if '--short' in sys.argv else 41
    rt.reset()
    if hook: hook[0](chunks)
    z=gen(emb,701,chunks)
    assert z.isfinite().all()
    vae=native_vae()
    frames=decode(vae,z,80 if chunks==7 else 480)
    path=folder/(method+'_'+variant+('_5s' if chunks==7 else ''))
    np.save(path.with_suffix('.npy'),np.concatenate(frames))
    save_video(path.with_suffix('.mp4'),frames)
    if callable(getattr(rt,'close',None)): rt.close()
main(sys.argv[1])
