import importlib.util,sys
from pathlib import Path
import torch
dest=Path(__file__).resolve().parents[2]/'reproduction/vbench_current'
from quota_counts import counts
from anchor_policy import install
from phase import set_phase
from local_experiments.video_runtime_v1 import pipeline_graph as base
from local_experiments.rollout_suite import generate_chunk

def attach(rt,engine,index,case):
    install(rt,'frame_fixed15')
    spec=importlib.util.spec_from_file_location(base.__package__+'.quota_ablation',(Path(__file__).with_name('mass_graph.py') if case=='mass' else dest/'graph.py'))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    mod.CACHED_LOGMASS=base.CACHED_LOGMASS
    bg=rt.batch_graph
    import numpy as np
    policy=np.load(dest/'policies.npz')[f'{index}_window_step_layer_head']
    clean=np.load(Path(__file__).resolve().parents[1]/'clean_forward/policies.npz')[f'{index}_window_step_layer_head']
    policy=np.concatenate([policy,clean],axis=1)
    schedule=np.empty((*policy.shape,2),dtype=np.int64)
    for c in range(2,41): schedule[c-2]=np.asarray(counts(min(432,72*(c+1))))[policy[c-2]]
    if case=='uniform':
        for i in range(len(schedule)):
            pages=min(432,216+72*i)
            total=(pages-24)//10
            current=max(1,round(0.42*total))
            schedule[i,...,0]=current
            schedule[i,...,1]=total-current
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
    bg.replay=audit
    def generate(emb,seed,chunks):
        zs=[]
        for c in range(chunks):
            if c>=6:
                if c==6: canonical=torch.cat([x['k'][:,:1536] for x in engine.cache],0)
                set_phase(rt,canonical,3*c-15)
            zs.append(generate_chunk(engine,emb,seed,c))
        return torch.cat(zs,1)
    return generate

REFERENCE = {0: 'ccacf3205dfcc50fe337df0abc45d5088c046c928d6ad82a720a9407c71d6fab', 1: '8a6ac39945be35c5bd904183cde7646efbe26bbeaca1d20f1360a27ba2d9a9a0'}

def verify_reference(z,scene,index):
    import hashlib
    assert index == 10
    data=z.detach().cpu().contiguous().view(torch.uint8).numpy()
    assert hashlib.sha256(data).hexdigest()==REFERENCE[scene], f'latent mismatch for scene {scene}'
