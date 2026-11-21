import json
import numpy as np
import torch
from local_experiments.high_sparse_experiment import prefix_quality

def schedule(rt, layers, heads):
    noisy = {f'{l}:0':[4,3,9] for l in (5,14,27,3,8,17,4)}
    rt.set_pset_schedule({**noisy, **{f'{l}:4':list(heads) for l in layers}})

def apply_selected(rt,folder):
    result=json.loads((folder/'scope.json').read_text())
    schedule(rt,result['layers'],result['heads'])

def search(rt,gen,emb,method,folder,native_vae,decode):
    from pathlib import Path
    import lpips
    vae=native_vae()
    emb=torch.load(folder/'embeddings/city_walk.pt',weights_only=True).cuda()
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    layers=list(range(1,30))
    heads=list(range(12))
    seeds=[701,702]
    def pixels(seed):
        rt.reset()
        return np.concatenate(decode(vae,gen(emb,seed,9),105))
    if method=='dense':
        for seed in seeds:
            np.save(folder/f'scope_ref{seed}.npy',pixels(seed))
        return
    refs={s:np.load(folder/f'scope_ref{s}.npy') for s in seeds}
    ledger=folder/'scope_scores.json'
    rows=json.loads(ledger.read_text()) if ledger.exists() else {}
    def score(name,ll,hh):
        if name not in rows:
            schedule(rt,ll,hh)
            rows[name]=[prefix_quality([pixels(s)],[refs[s]],metric)['lpips'] for s in seeds]
            ledger.write_text(json.dumps(rows))
        return np.mean(rows[name])
    off=score('off',[],[])
    ls=sorted(layers,key=lambda l:(score(f'layer{l}',[l],heads),l))[:7]
    hs=sorted(heads,key=lambda h:(score(f'head{h}',layers,[h]),h))[:3]
    combined=score('intersection',ls,hs)
    enabled=combined<off
    (folder/'scope.json').write_text(json.dumps(dict(layers=ls if enabled else [],heads=hs if enabled else [],candidate_layers=ls,candidate_heads=hs,off=off,combined=combined)))
