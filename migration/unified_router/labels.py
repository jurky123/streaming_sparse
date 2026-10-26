"""Stream exact action supervision; never expose LabelStore to online routing."""
import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import torch
from .common import config, load_raw, event_key, save, atomic_json, update, paired_seed, sha
from .core import build, scores, initial_mask, candidates, features, LabelStore
from .fit import eig


GROUPS = {'K32-Set': 'K32', 'V8-Set': 'V8', 'V32-NoSet': 'V32', 'V32-Set': 'V32'}
TRAIN_BETAS = [-.5, .5, 1.5]


@lru_cache(maxsize=8)
def feature_namespace(root):
    return json.loads((Path(root)/'experiment.json').read_text()).get('feature_namespace')


def feature_path(root, cache, row, layer, beta):
    # New budget runs own their features; never overwrite the earlier experiment.
    namespace = feature_namespace(str(root))
    directory = Path(cache)/'features'
    if namespace:
        directory = directory/namespace
    tag = f'{beta:+.4f}' if namespace else f'{beta:+.1f}'
    return directory/f'beta_{tag}'/event_key(row)/f'layer{layer:02d}.pt'


def train_betas(m):
    return m.get('train_betas', TRAIN_BETAS)


def parameters(root, layer):
    d = torch.load(root/f'parameters/layer{layer:02d}.pt', map_location='cpu', weights_only=True)
    return d, {k:v.cuda() for k,v in d['parameters'].items()}


def projection(root, m, layer):
    final = root/f'action_basis/layer{layer:02d}.pt'
    if final.exists():
        return
    initial, p = parameters(root, layer)
    covariance = torch.zeros(12, 128, 128, device='cuda', dtype=torch.float64)
    energy = torch.zeros(12, 4, device='cuda', dtype=torch.float64)
    counts = torch.zeros(4, device='cuda', dtype=torch.float64)
    gains = {(h,t):[] for h in range(12) for t in range(4)}
    train = [r for r in m['rows'] if r['split']=='train' and r['history_tokens']]
    chosen = [next(r for r in train if r['prompt_id']==pid and r['generation_seed']==i%2
        and r['step']==i%4 and r['history_tokens']==[4680,9360,14040][i%3])
        for i,pid in enumerate(m['splits']['train'])]
    for index, row in enumerate(chosen, 1):
        data = load_raw(root,row,layer)
        data['q'] = data['q'][:, :32]
        sketch = build(**{k:data[k] for k in ['kh','vh','kc','vc']}, parameters=p, need_value=False)
        score,_ = scores(data['q'],sketch,content=None)
        truth = LabelStore(**data)
        e = truth.target.square().sum(-1).mean(-1)
        energy[:,row['step']] += e
        counts[row['step']] += 1
        for beta in train_betas(m):
            mask = initial_mask(score,sketch,beta)
            actions = candidates(score,mask,sketch.origins,beta,seed=16)
            label,_,cov = truth.action_labels(mask,actions,covariance=True)
            covariance += cov/(len(chosen)*len(train_betas(m)))
            for h in range(12):
                valid = actions['valid'].reshape(12,32,81)[h]
                for typ in [1,2,3]:
                    values = label[h,:,actions['types']==typ]
                    valid_t = valid[:,actions['types']==typ]
                    values = values[valid_t].abs()
                    if values.numel():
                        gains[h,row['step']].append(values[:256].cpu())
        update(root/'P2_projection'/f'layer{layer:02d}.json',status='fitting',completed=index,total=len(chosen))
    _, u = eig(covariance)
    energy = (energy/counts.clamp_min(1)[None]).clamp_min(1e-8).float()
    kappa = torch.ones(12,4)
    for (h,t), values in gains.items():
        if values:
            kappa[h,t] = (torch.cat(values).median()/energy[h,t].cpu()).clamp_min(1e-6)
    p['value_basis'] = u[...,:32]
    save(final,dict(parameters={k:v.cpu()for k,v in p.items()},energy_scale=energy.cpu(),gain_scale=kappa,
        layer=layer,train_sample_ids=[r['sample_id']for r in chosen],
        type_balanced_action_covariance=True,basis_dim=32,foundation_updates=initial['foundation_updates']))
    update(root/'P2_projection'/f'layer{layer:02d}.json',status='completed',basis_dim=32)


def generate(root,m,layer):
    state=torch.load(root/f'action_basis/layer{layer:02d}.pt',map_location='cpu',weights_only=True)
    p={k:v.cuda()for k,v in state['parameters'].items()}
    rows=[r for r in m['rows']if r['history_tokens']]
    for index,row in enumerate(rows,1):
        betas=train_betas(m) if row['split']=='train'else m['beta_grid']
        paths=[feature_path(root,m['cache'],row,layer,beta)for beta in betas]
        if all(path.exists()for path in paths):
            continue
        data=load_raw(root,row,layer)
        sketch=build(**{k:data[k]for k in ['kh','vh','kc','vc']},parameters=p)
        score,cv32=scores(data['q'],sketch,content='V32')
        _,ck32=scores(data['q'],sketch,content='K32')
        truth=LabelStore(**data)
        for beta,path in zip(betas,paths):
            if path.exists():continue
            mask=initial_mask(score,sketch,beta)
            actions=candidates(score,mask,sketch.origins,beta,seed=16)
            labels,error,_=truth.action_labels(mask,actions)
            geometry={};key=None
            for kind in m.get('groups',GROUPS):
                content=ck32 if kind=='K32-Set'else cv32[...,:8]if kind=='V8-Set'else cv32
                g,k=features(data['q'],sketch,score,mask,actions,beta,row['step'],content,no_set=kind=='V32-NoSet')
                geometry[kind]=g.reshape(12,len(data['q'][0]),81,8).cpu()
                if key is None:key=k.reshape(12,len(data['q'][0]),81,16).cpu()
            i,j=actions['remove'],actions['add'];costs=sketch.costs[None].expand(len(i),-1);origin=sketch.origins[None].expand_as(costs)
            def gather(x,ix):return x.gather(-1,ix.clamp_min(0))*(ix>=0)
            dh=gather(costs*(1-origin),j)-gather(costs*(1-origin),i)
            dc=gather(costs*origin,j)-gather(costs*origin,i)
            base_h=(mask*sketch.costs[None,None]*(1-sketch.origins)[None,None]).sum(-1)
            base_c=(mask*sketch.costs[None,None]*sketch.origins[None,None]).sum(-1)
            save(path,dict(key=key,geometry=geometry,gain=labels.cpu(),base_error=error.cpu(),
                energy=truth.target.square().sum(-1).float().cpu(),valid=actions['valid'].reshape(12,len(data['q'][0]),81).cpu(),
                types=actions['types'].cpu(),delta_H=dh.reshape_as(labels).cpu(),delta_C=dc.reshape_as(labels).cpu(),
                base_H=base_h.cpu(),base_C=base_c.cpu(),history_tokens=row['history_tokens'],current_tokens=4680,
                sample_id=row['sample_id'],split=row['split'],prompt_id=row['prompt_id'],step=row['step'],beta=beta,
                label_kind='exact original128-d hard actions; offline only',key_basis_hash=sha(root/f'action_basis/layer{layer:02d}.pt')))
        update(root/'P2_features'/f'layer{layer:02d}.json',status='generating',completed=index,total=len(rows))
    # Fixed feature normalization uses16 Train prompts only, never Val.
    groups=m.get('groups',GROUPS)
    sums={k:torch.zeros(12,24,dtype=torch.float64)for k in groups};sq={k:torch.zeros(12,24,dtype=torch.float64)for k in groups};counts=0
    for pid in m['splits']['train']:
        row=next(r for r in rows if r['prompt_id']==pid and r['split']=='train')
        for beta in train_betas(m):
            d=torch.load(feature_path(root,m['cache'],row,layer,beta),map_location='cpu',weights_only=True,mmap=True)
            for kind in groups:
                x=torch.cat((d['geometry'][kind],d['key']),-1).double().flatten(1,2)
                sums[kind]+=x.sum(1);sq[kind]+=x.square().sum(1)
            counts+=x.shape[1]
    normal={k:dict(mean=(sums[k]/counts).float(),std=(sq[k]/counts-(sums[k]/counts).square()).clamp_min(0).sqrt().clamp_min(1e-3).float())for k in groups}
    save(root/f'normalization/layer{layer:02d}.pt',normal)
    update(root/'P2_features'/f'layer{layer:02d}.json',status='completed',events=len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--projection',action='store_true');a=p.parse_args()
    root,m=config(a.directory);paired_seed(40);torch.cuda.set_device(0)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    for layer in m['layers'][rank::world]:(projection if a.projection else generate)(root,m,layer)


if __name__=='__main__':main()
