import importlib.util,json,sys,time
from pathlib import Path
import torch
from scan import ROOT,GRID,counts
from phase import set_phase
from anchor_policy import install
from local_experiments.video_runtime_v1 import pipeline_graph as base
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from local_experiments.stream_eval_common import decode,native_vae,save_video

@torch.no_grad()
def run(kind):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=config('intersection',profile=True,local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine();engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg);install(rt,'frame_fixed15')
    if kind=='dense_no_route': rt.sparse=False
    if kind=='refdense':
        from local_experiments.video_runtime_v1 import legacy_components
        legacy_components.attention=lambda q,k,v,*args,**kw:engine.dense(q,k,v)
    emb=torch.load(ROOT.parent.parent/'embeddings/long_0.pt',weights_only=True).cuda()
    info={}
    for name,n in [('warmup',7),('wall',41),('profile',41)]:
        rt.reset();rt.timings.enabled=name=='profile'
        torch.cuda.synchronize();t=time.monotonic();zs=[]
        for c in range(n):
            if c>=6:
                if c==6: canonical=torch.cat([x['k'][:,:1536] for x in engine.cache],0)
                with rt.timings.stage('anchor_rebase'):set_phase(rt,canonical,3*c-15)
            zs.append(generate_chunk(engine,emb,701,c))
        z=torch.cat(zs,1);torch.cuda.synchronize()
        info[name]=time.monotonic()-t
        assert z.isfinite().all()
        print(name,info[name],flush=True)
    info.update(spec=kind,stages=rt.timings.finish())
    (ROOT/f'{kind}.json').write_text(json.dumps(info,indent=2))
    save_video(ROOT/f'{kind}.mp4',decode(native_vae(),z,480))
run(sys.argv[1])
