import importlib.util,json,sys,time
from pathlib import Path
import os
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','4')
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
def run(index,scene,variant,repeat=None):
    dest=Path(__file__).resolve().parent.parent/'conditional_budget_v1'
    seed=701+scene
    label=f"{scene}_{index}_{variant}"
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location(base.__package__+'.quota_ablation',dest/'graph.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    cfg=config('intersection',profile=True,local_assets='/tmp/streaming_sparse-0/runtime')
    if variant in ('trained','C_only'):
        cfg.checkpoint=str(dest.parent/'rebased_retrain_v1'/f'budget_{11 if index==10 else 21}/summary.pt')
    if variant in ('trained','reader_only'):
        cfg.reader_checkpoint=str(dest.parent/'rebased_retrain_v1'/f'budget_{11 if index==10 else 21}/reader.pt')
    if variant=='refit':
        cfg.checkpoint=str(dest/f'budget_{index}/summary.pt')
        cfg.reader_checkpoint=str(dest/f'budget_{index}/reader.pt')
    configure(cfg)
    engine=Engine();engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg);install(rt,'frame_fixed15')
    if variant=='dense':
        rt.sparse=False
    mod.CACHED_LOGMASS=base.CACHED_LOGMASS
    bg=rt.batch_graph
    import numpy as np
    policy=np.load(dest/'policies.npz')[f'{index}_window_step_layer_head']
    schedule=np.empty((*policy.shape,2),dtype=np.int64)
    for c in range(2,41): schedule[c-2]=np.asarray(counts(min(432,72*(c+1))))[policy[c-2]]
    totals=schedule.sum(-1)
    assert (totals==totals[...,0,None]).all()
    slots=torch.from_numpy(schedule[...,0].copy()).cuda()
    bg.current_slots=torch.zeros(12,device='cuda',dtype=torch.long)
    def get(*args,**kwargs):
        c,l,step=[engine.current[k] for k in ('chunk','layer','step')]
        bg.n=int(totals[c-2,step,l-1,0])
        bg.current_slots.copy_(slots[c-2,step,l-1])
        return mod.BatchGraphs.get(bg,*args,**kwargs)
    bg.get=get
    bg.compute=torch.compile(mod.BatchGraphs.compute.__get__(bg,type(bg)),fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    replay=bg.replay
    seen=set()
    def audit(e,q,own):
        ids,mask,risk,stats=replay(e,q,own)
        key=(mask.size(-1),bg.n,e['full_pset'])
        if key not in seen:
            assert mask[...,:24].all() and (mask[...,-72:].sum(-1)==bg.current_slots[:,None]).all()
            assert (ids[...,0]==own).all() and mask.gather(-1,ids.long()).all() and (ids.sort().values.diff()>0).all()
            assert ids.shape[-1]==24+bg.n
            seen.add(key)
        return ids,mask,risk,stats
    emb=torch.load(ROOT.parent.parent/'embeddings'/f'long_{scene}.pt',weights_only=True).cuda()
    from local_experiments.video_runtime_v1 import legacy_components
    from collections import defaultdict
    original_attention=legacy_components.attention
    events=defaultdict(list)
    def timed_attention(q,k,v,s,basis,ids,workspace):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        out=original_attention(q,k,v,s,basis,ids,workspace)
        b.record()
        events[(s.mean.shape[1],ids.shape[-1])].append((a,b))
        return out
    info={}
    phases=[('warmup',7),('wall',41),('wall2',41),('wall3',41)]
    if repeat is None:phases.append(('profile',41))
    for name,n in phases:
        legacy_components.attention=timed_attention if name=='profile' else original_attention
        rt.reset();rt.timings.enabled=name=='profile'
        bg.replay=audit if name=='warmup' else replay
        torch.cuda.synchronize();t=time.monotonic();zs=[]
        for c in range(n):
            if c>=6:
                if c==6: canonical=torch.cat([x['k'][:,:1536] for x in engine.cache],0)
                with rt.timings.stage('anchor_rebase'):set_phase(rt,canonical,3*c-15)
            zs.append(generate_chunk(engine,emb,seed,c))
        z=torch.cat(zs,1);torch.cuda.synchronize()
        info[name]=time.monotonic()-t
        assert z.isfinite().all() and (variant=='dense' or len(seen)>0)
        print(name,info[name],flush=True)
    info['fusion_device_ms']=[dict(pages=p,exact=k,calls=len(ev),ms=sum(a.elapsed_time(b) for a,b in ev)) for (p,k),ev in events.items()]
    info.update(budget=index,shapes=len(seen),stages=rt.timings.finish())
    if repeat is not None:
        (Path(__file__).parent/f'table_{index}_{scene}.json').write_text(json.dumps(info,indent=2))
        return
    torch.save(z.cpu(),dest/(label+'.pt'))
    (dest/(label+'.json')).write_text(json.dumps(info,indent=2))
    save_video(dest/(label+'.mp4'),decode(native_vae(),z,480))
run(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],*sys.argv[4:])
