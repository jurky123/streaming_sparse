"""Repeat the historical scope ranking while holding the current quotas fixed."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch

out=Path(__file__).resolve().parent
repo=out.parents[1]
clean=out.parent/'clean_forward'
sys.path[:0]=[str(clean),str(repo/'reproduction/vbench_current')]
from clean_policy import attach
from scope import apply_selected
from local_experiments.video_runtime_v1 import scope_experiment as history
from local_experiments.broad_common import Engine
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,load_weights,make_runtime
from local_experiments.high_sparse_experiment import prefix_quality
from local_experiments.stream_eval_common import native_vae,decode

history.OUT=out
def scope(rt,job):
    if job['name']=='old':
        apply_selected(rt,clean)
        return
    schedule=job.get('schedule')
    if schedule is None:
        schedule={} if job['name']=='base' else {
            f'{l}:{s}':job.get('heads',list(range(12)))
            for l in job.get('layers',range(1,30)) for s in job.get('steps',range(4))}
    rt.set_pset_schedule(schedule)

@torch.no_grad()
def run(phase,seed):
    import lpips
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.cache_size_limit=512
    torch._dynamo.config.accumulated_cache_size_limit=4096
    cfg=config('intersection_fast',local_assets=repo)
    configure(cfg)
    torch._dynamo.config.cache_size_limit=2048
    torch._dynamo.config.accumulated_cache_size_limit=8192
    engine=Engine()
    for b in engine.wrapper.model.blocks:b.self_attn.sink_size=1
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    rt.sparse_clean=True
    gen=attach(rt,engine,10)
    torch._dynamo.config.recompile_limit=2048
    torch._dynamo.config.accumulated_recompile_limit=8192
    vae=native_vae()
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    jobs=json.loads((out/('selected_manifest.json' if phase=='holdout' else phase+'_manifest.json')).read_text())
    if phase=='selected':jobs+=[dict(name='old',seed=seed)]
    if phase=='holdout':
        job=next(j for j in jobs if j['name']=='intersection')
        jobs=[dict(name='base'),dict(name='old'),job]
    for prompt in (['city_walk','short_721'] if phase=='holdout' else ['city_walk']):
        embpath=clean/'embeddings/city_walk.pt' if prompt=='city_walk' else repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt'
        emb=torch.load(embpath,weights_only=True).cuda()
        ref=np.load(clean/f'scope_ref{seed}.npy',mmap_mode='r') if phase!='holdout' else None
        for job in jobs:
            if phase!='holdout' and seed!=job['seed']:continue
            name=job['name']
            folder=out/phase/name/(f'{prompt}_s{seed}' if phase=='holdout' else f's{seed}')
            path=folder/'result.json'
            if path.exists() and 'generation_seconds' in json.loads(path.read_text()):continue
            scope(rt,job)
            rt.reset()
            length=7 if phase=='holdout' else 9
            gen(emb,seed,length)
            rt.reset()
            torch.cuda.synchronize()
            start=time.perf_counter()
            z=gen(emb,seed,length)
            torch.cuda.synchronize()
            generation=time.perf_counter()-start
            assert z.isfinite().all()
            pixels=decode(vae,z,80 if phase=='holdout' else 105)
            if phase=='holdout':
                reference=np.load(clean/'module_sensitivity'/f'ref_{prompt}_{seed}.npy',mmap_mode='r')
                # Match the ablation's whole 80-frame LPIPS with the same scorer.
                def pad(x):return [np.concatenate([x[:21],x])]
                quality=prefix_quality(pad(np.concatenate(pixels)),pad(reference),metric)
            else:quality=prefix_quality(pixels,[ref],metric)
            history.write(path,dict(job=job,quality=quality,seed=seed,prompt=prompt,generation_seconds=generation))
            print(phase,name,seed,prompt,quality['lpips'],flush=True)

if __name__=='__main__':
    if sys.argv[1]=='init':history.initial()
    elif sys.argv[1]=='select':
        history.select()
        f=out/'selection.json'
        result=json.loads(f.read_text())
        result['rounding']='floor: 7/29 layers, 3/12 heads, 1/4 noisy steps; layer0 dense, clean step4 sparse with correction off'
        history.write(f,result)
    else:run(sys.argv[1],int(sys.argv[2]))
