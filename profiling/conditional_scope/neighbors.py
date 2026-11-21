"""Fixed-budget single swaps and frozen paired validation, following intersection_search_v1."""
import json,sys,time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
import numpy as np
import torch
from scan import out,clean,repo,scope,history,config,configure,Engine,make_runtime,load_weights,attach,native_vae,decode,prefix_quality
from local_experiments.video_runtime_v1.intersection_search import neighbors,normalize,policy_id
from local_experiments.video_runtime_v1.intersection_validation import CASES,SEEDS
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.backbone_ops import install
from phase import set_phase
write=history.write
root=out/'neighbors'
root.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def initial():
    return normalize(next(j for j in read(out/'selected_manifest.json') if j['name']=='intersection'))
def records(p):
    directory=out/'selected/intersection' if p==initial() else root/'scores'/policy_id(p)
    quality=[read(directory/f's{s}/result.json')['quality']['lpips'] for s in [701,702]]
    return dict(policy=p,lpips=mean(quality),seeds=quality)
def manifest(r):
    center=initial() if r==1 else read(root/'winner1.json')['policy']
    write(root/f'round{r}.json',[x['policy'] for x in neighbors(center)])
def select(r):
    center=initial() if r==1 else read(root/'winner1.json')['policy']
    best=min([records(p) for p in [center]+read(root/f'round{r}.json')],key=lambda x:(x['lpips'],policy_id(x['policy'])))
    if records(center)['lpips']-best['lpips']<=1e-6:best=records(center)
    write(root/f'winner{r}.json',best)
def freeze():
    orig=records(initial())
    seen={policy_id(initial()):initial()}
    for f in [root/'round1.json',root/'round2.json']:
        if f.exists():seen.update((policy_id(p),p) for p in read(f))
    ranks=sorted(map(records,seen.values()),key=lambda x:(x['lpips'],policy_id(x['policy'])))
    balanced=next(r for r in ranks if all(a<=b+1e-6 for a,b in zip(r['seeds'],orig['seeds'])))
    write(root/'frozen.json',dict(single_axis=orig,search_mean=ranks[0],search_balanced=balanced,unique=len(seen)))
def embpath(p):
    prev=clean/'embeddings'/f'{p}.pt'
    return prev if prev.exists() else root/f'{p}.pt'
@torch.inference_mode()
def embed():
    from local_experiments.broad_common import PROMPTS
    from utils.wan_wrapper import WanTextEncoder
    model=WanTextEncoder().to(device='cuda',dtype=torch.bfloat16)
    prompts=dict(sum(PROMPTS.values(),[]))
    for p in CASES:
        if not embpath(p).exists():torch.save(model([prompts[p]])['prompt_embeds'].cpu(),embpath(p))
@torch.no_grad()
def worker(mode,idx):
    import lpips
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('intersection_fast',local_assets=repo)
    if mode=='dense':cfg.mode='dense_native'
    configure(cfg)
    engine=Engine()
    for b in engine.wrapper.model.blocks:b.self_attn.sink_size=1
    if mode=='dense':
        install(engine,cfg)
        rt=SimpleNamespace(engine=engine,stores={},reset=engine.reset)
        def gen(emb,s,n):
            parts=[]
            for c in range(n):
                if c==6:canonical=torch.cat([cache['k'][:,:1536] for cache in engine.cache])
                if c>=6:set_phase(rt,canonical,3*c-15)
                parts.append(generate_chunk(engine,emb,s,c))
            return torch.cat(parts,1)
    else:
        rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
        rt.sparse_clean=True
        gen=attach(rt,engine,10)
    torch._dynamo.config.recompile_limit=2048
    torch._dynamo.config.accumulated_recompile_limit=8192
    vae=native_vae()
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    if mode.startswith('round'):
        seed=701+idx
        ref=np.load(clean/f'scope_ref{seed}.npy',mmap_mode='r')
        emb=torch.load(embpath('city_walk'),weights_only=True).cuda()
        for p in read(root/f'{mode}.json'):
            path=root/'scores'/policy_id(p)/f's{seed}/result.json'
            if p==initial() or path.exists():continue
            scope(rt,dict(name='candidate',**p))
            rt.reset()
            z=gen(emb,seed,9)
            assert z.isfinite().all()
            q=prefix_quality(decode(vae,z,105),[ref],metric)
            write(path,dict(quality=q,policy=p,seed=seed))
        return
    if mode=='validate':
        frozen=read(root/'frozen.json')
        specs={n:dict(name=n,**frozen[n]['policy']) for n in ['single_axis','search_mean','search_balanced']}
        specs.update(base={'name':'base'},old={'name':'old'})
        for j in read(out/'selected_manifest.json'):
            if j['name'].startswith('top_'):specs[j['name']]=j
    for i,(p,seed) in enumerate((p,s) for p in CASES for s in SEEDS):
        if i%2!=idx:continue
        dest=root/'validation'/f'{p}_s{seed}'
        dest.mkdir(parents=True,exist_ok=True)
        emb=torch.load(embpath(p),weights_only=True).cuda()
        refpath=dest/'dense.npy'
        if mode=='dense':
            if not refpath.exists():
                rt.reset()
                np.save(refpath,np.concatenate(decode(vae,gen(emb,seed,9),105)))
            continue
        ref=np.load(refpath,mmap_mode='r')
        names=list(specs)
        for n in names[i%len(names):]+names[:i%len(names)]:
            path=dest/(n+'.json')
            if path.exists():continue
            scope(rt,specs[n])
            # Each timed pass follows a full warmup of the exact scope.
            rt.reset();gen(emb,seed,9);rt.reset()
            torch.cuda.synchronize();t=time.perf_counter()
            z=gen(emb,seed,9);torch.cuda.synchronize();sec=time.perf_counter()-t
            assert z.isfinite().all()
            write(path,dict(quality=prefix_quality(decode(vae,z,105),[ref],metric),seconds=sec,prompt=p,seed=seed,scope=specs[n]))
if __name__=='__main__':
    m=sys.argv[1]
    if m=='embed':embed()
    elif m=='manifest':manifest(int(sys.argv[2]))
    elif m=='select':select(int(sys.argv[2]))
    elif m=='freeze':freeze()
    else:worker(m,int(sys.argv[2]))
