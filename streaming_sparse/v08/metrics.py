"""Input-equal, then prompt-equal NMSE and fully paired prompt bootstrap."""
from collections import defaultdict
import numpy as np


def summarize(rows):
    if not rows:raise ValueError('Empty replay evaluation')
    grouped=defaultdict(list)
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate replay input')
    for r in rows:grouped[r['prompt_id']].append(r)
    prompt={k:float(np.mean([r['nmse'] for r in v])) for k,v in grouped.items()}
    values=np.asarray([r['nmse'] for r in rows])
    out=dict(prompt_equal_error=float(np.mean(list(prompt.values()))),prompts=prompt,
        micro_error=sum(r['sse'] for r in rows)/(sum(r['energy'] for r in rows)+1e-8),inputs=len(rows),prompt_count=len(prompt),
        input_error=dict(mean=float(values.mean()),median=float(np.median(values)),p90=float(np.quantile(values,.9)),p99=float(np.quantile(values,.99))))
    for field in ('generation_seed','history_tokens','step'):
        out[field]={}
        for value in sorted({r[field] for r in rows}):
            part=[r for r in rows if r[field]==value];pg=defaultdict(list)
            for r in part:pg[r['prompt_id']].append(r['nmse'])
            out[field][str(value)]=float(np.mean([np.mean(v) for v in pg.values()]))
    return out


def factorial(models):
    """models[(router_seed,group)] is a summary; paired seeds never act as prompts."""
    groups=('F-K','F-V','A-K','A-V','J-K','J-V');seeds=(0,1,2)
    if set(models)!={(s,g) for s in seeds for g in groups}:raise ValueError('All6 groups x3 paired seeds are required')
    prompts=sorted(models[(0,'F-K')]['prompts'])
    for x in models.values():
        if sorted(x['prompts'])!=prompts:raise ValueError('Unpaired prompt set')
    fingerprints={x.get('dataset_fingerprint') for x in models.values()}
    if len(fingerprints)>1:raise ValueError('Unpaired replay input manifests')
    matrix={g:np.asarray([[models[s,g]['prompts'][p] for p in prompts] for s in seeds]) for g in groups}
    from .common import protocol
    p=protocol()['endpoint'];rng=np.random.default_rng(p['bootstrap_seed'])
    draw=rng.integers(0,len(prompts),size=(p['bootstrap_repeats'],len(prompts)))
    gf=matrix['F-K']-matrix['F-V'];gj=matrix['J-K']-matrix['J-V'];interaction=gj-gf
    result={}
    for name,value in {'G_F':gf,'G_J':gj,'I':interaction,'J_V_vs_A_V':matrix['A-V']-matrix['J-V']}.items():
        boot=value.mean(0)[draw].mean(-1)
        result[name]=dict(mean=float(value.mean()),per_seed=value.mean(1).tolist(),
            prompt_paired_95CI=np.quantile(boot,[.025,.975]).tolist(),
            positive_training_seeds=int((value.mean(1)>0).sum()))
    rel=(matrix['J-K'].mean()-matrix['J-V'].mean())/matrix['J-K'].mean()
    result['joint_V_relative_gain']=float(rel)
    result['investment_gate']=bool(rel>=p['investment_relative_gain'] and result['G_J']['prompt_paired_95CI'][0]>0
        and result['G_J']['positive_training_seeds']>=p['same_direction_seeds_min'])
    result['caveat']='I and its CI reported separately; positive G_J does not alone establish enlarged advantage. Prompt is bootstrap unit.'
    return result
