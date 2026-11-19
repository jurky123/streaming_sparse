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
def run(index):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location(base.__package__+'.quota_ablation',ROOT/'quota_graph.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    cfg=config('intersection',profile=True,local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine();engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg);install(rt,'frame_fixed15')
    mod.CACHED_LOGMASS=base.CACHED_LOGMASS
    bg=rt.batch_graph
    bg.quota={p:counts(p)[index] for p in (216,288,360,432)}
    bg.compute=torch.compile(mod.BatchGraphs.compute.__get__(bg,type(bg)),fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    replay=bg.replay
    seen=set()
    def audit(e,q,own):
        ids,mask,risk,stats=replay(e,q,own);p=mask.shape[-1]
        key=(p,e['full_pset'])
        if key not in seen:
            a,b=bg.quota[p]
            assert mask[...,:24].all() and (mask[...,-72:].sum(-1)==a).all() and (mask[...,24:-72].sum(-1)==b).all()
            assert (ids[...,0]==own).all() and mask.gather(-1,ids.long()).all()
            assert (ids.sort().values.diff()>0).all() and ids.shape[-1]==24+a+b
            seen.add(key)
        return ids,mask,risk,stats
    emb=torch.load(ROOT.parent.parent/'embeddings/long_0.pt',weights_only=True).cuda()
    info={}
    for name,n in [('warmup',7),('wall',41),('profile',41)]:
        rt.reset();rt.timings.enabled=name=='profile'
        bg.replay=audit if name=='warmup' else replay
        torch.cuda.synchronize();t=time.monotonic();zs=[]
        for c in range(n):
            if c>=6:
                if c==6: canonical=torch.cat([x['k'][:,:1536] for x in engine.cache],0)
                with rt.timings.stage('anchor_rebase'):set_phase(rt,canonical,3*c-15)
            zs.append(generate_chunk(engine,emb,701,c))
        z=torch.cat(zs,1);torch.cuda.synchronize()
        info[name]=time.monotonic()-t
        assert z.isfinite().all() and len(seen)==8
        print(name,info[name],flush=True)
    info.update(spec=GRID[index],counts=bg.quota,stages=rt.timings.finish())
    (ROOT/f'closed_{index}.json').write_text(json.dumps(info,indent=2))
    save_video(ROOT/f'closed_{index}.mp4',decode(native_vae(),z,480))
run(int(sys.argv[1]))
