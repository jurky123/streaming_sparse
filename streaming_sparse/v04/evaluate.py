"""Paired full-data evaluation with durable snapshot commits and Q diagnostics."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import shutil
import statistics as py_stats
import torch
from .data import SnapshotData
from .identity import source_identity,file_sha
from .model import load_pair,partition_hash
from .metrics import summarize
from ..v03.evaluate import statistics
from ..v03.metrics import error_parts,correlation,partition_diagnostics
from .paths import protect_output
from ..capture.dataset_io import query_slices,atomic_save
from ..evaluation.baselines import mass_selection
from ..v02.evaluation_journal import atomic_json


def error(output,target):
    se,en=error_parts(output,target)
    return dict(sse=float(se),energy=float(en),error=float(se/(en+1e-8)))


def jaccard(a,b):
    union=int((a|b).sum())
    return float((a&b).sum())/union if union else 1.


@torch.no_grad()
def diagnostics(q,event,state,stat,routes):
    keys=torch.cat((event['k_hist'],event['k_current'])).float()
    logp=(q.float()@keys.T/math.sqrt(128)).log_softmax(-1)
    entropy=float(-(logp.exp()*logp).sum(-1).mean())
    hz=stat.z/stat.z.sum(-1,keepdim=True).clamp_min(1e-30)
    concentration=float(hz.square().sum(-1).mean())
    norm=torch.nn.functional.normalize(state['z'].float(),dim=-1)
    b=len(norm)
    redundancy=float((norm.sum(0).square().sum()-b)/max(b*(b-1),1))
    masses=(stat.z/(stat.z.sum(-1)+stat.z_current)[:,None].clamp_min(1e-30)).sum(0)
    scores={name:dict(mean=float(r['scores'].mean()),std=float(r['scores'].std(unbiased=False)),
                     mass_correlation=correlation(r['scores'],masses),
                     entropy=float(-(r['scores'].softmax(-1)*r['scores'].log_softmax(-1)).sum()))
            for name,r in routes.items()}
    return dict(query_entropy_all_raw_KV=entropy,history_block_mass_concentration=concentration,
                z_mean_pair_cosine_redundancy_proxy=redundancy,score_distribution=scores)


@torch.no_grad()
def evaluate_snapshot(model,events,keeps,save_partition=None,seen_tiles=()):
    e0=events[0]
    state=model.precompute(e0['k_hist'],e0['v_hist'],head_id=e0['head_id'],layer_id=e0['layer_id'])
    assert partition_hash(state['hard'])==state['partition_hash']
    if save_partition is not None:
        atomic_save(dict(labels=state['hard'].argmax(-1).to(device='cpu',dtype=torch.int16),
                         sizes=state['hard'].sum(0).int().cpu(),partition_hash=state['partition_hash'],
                         original_token_index_definition='row index of immutable raw history KV',
                         token_count=len(state['hard'])),save_partition)
    rows=[]
    sensitivity=[]
    costs=state['hard'].sum(0)
    for event in events:
        masks=defaultdict(list)
        tile_errors=defaultdict(list)
        for tile,selection in enumerate(query_slices(event,16)):
            q=event['q'][selection]
            stat=statistics(q,event,state['assignment'])
            for keep in keeps:
                routes=model.route(q,state,keep)
                budget=math.floor(keep*len(state['hard']))
                mass=mass_selection(stat,costs,budget)
                all_masks={name:r['hard_mask'] for name,r in routes.items()}
                all_masks['Mass']=mass
                tokens={name:(state['hard']@mask>0) for name,mask in all_masks.items()}
                errors={name:error(stat.output(mask),event['o_dense'][selection]) for name,mask in all_masks.items()}
                retained={name:int((costs*mask).sum()) for name,mask in all_masks.items()}
                if any(count>budget for count in retained.values()):
                    raise AssertionError('Token budget exceeded')
                row=dict(cache_version=event['cache_version'],prompt_id=event['prompt_id'],seed=event['seed'],
                    layer_id=event['layer_id'],head_id=event['head_id'],step_id=event['step_id'],
                    history_tokens=len(state['hard']),tile_id=tile,
                    query_tile_seen_in_training=tile in seen_tiles,
                    query_indices=event['query_indices'][selection].cpu().tolist(),keep=keep,
                    partition_hash_A=state['partition_hash'],partition_hash_B=state['partition_hash'],
                    aggregation_identity='one shared z tensor consumed by both branches',
                    block_count=len(costs),budget=budget,retained=retained,errors=errors,
                    actual_retained_ratio={name:n/len(state['hard']) for name,n in retained.items()},
                    selected_block_ids={name:torch.where(mask>0)[0].cpu().tolist() for name,mask in all_masks.items()},
                    A_B_token_jaccard=jaccard(tokens['A'],tokens['B']),
                    **diagnostics(q,event,state,stat,routes))
                rows.append(row)
                for name in ('A','B'):
                    masks[(keep,name)].append(tokens[name])
                    tile_errors[(keep,name)].append(errors[name]['error'])
        for (keep,name),values in masks.items():
            similarities=[jaccard(a,b) for a,b in combinations(values,2)]
            sensitivity.append(dict(cache_version=event['cache_version'],step_id=event['step_id'],keep=keep,
                layer_id=event['layer_id'],head_id=event['head_id'],history_tokens=len(state['hard']),
                prompt_id=event['prompt_id'],seed=event['seed'],
                arm=name,tile_count=len(values),pair_count=len(similarities),
                mean_tile_mask_jaccard=py_stats.mean(similarities) if similarities else 1.,
                min_tile_mask_jaccard=min(similarities) if similarities else 1.,
                tile_attention_errors=tile_errors[(keep,name)],
                interpretation='Mask differences are descriptive, not an optimization objective'))
    return dict(rows=rows,query_sensitivity=sensitivity,partition=partition_diagnostics(state))


def evaluate(model,data,output,keeps=(.3,.2,.1),checkpoint_sha=None,exposure=None,training_identity=None):
    if checkpoint_sha is None:
        raise ValueError('Evaluation requires the SHA256 of its exact paired checkpoint')
    output=protect_output(output)
    signature=dict(source_sha256=source_identity(),data_fingerprint=data.fingerprint(),split=data.split,
                   checkpoint_sha256=checkpoint_sha,keeps=list(keeps),snapshots=sorted(data.groups))
    meta=output/'identity.json'
    if meta.exists():
        if json.loads(meta.read_text())!=signature:
            raise ValueError('Evaluation identity mismatch')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Nonempty evaluation directory without identity')
    else:
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(meta,signature)
    committed=output/'snapshot_commits'
    committed.mkdir(exist_ok=True)
    partitions=output/'partitions'
    partitions.mkdir(exist_ok=True)
    for i,version in enumerate(sorted(data.groups)):
        done=committed/f'{version}.json'
        if done.exists():
            value=json.loads(done.read_text())
            expected=4*8*len(keeps)
            if len(value['rows'])!=expected or any(r['cache_version']!=version for r in value['rows']):
                raise ValueError('Corrupt snapshot commit')
            if not (partitions/f'{version}.pt').is_file():
                raise ValueError('Snapshot commit is missing its partition evidence')
            continue
        count=exposure['visits'].get(version,0) if exposure else 0
        offset=exposure['offset'].get(version,0) if exposure else 0
        seen={(offset+i)%8 for i in range(min(count,8))}
        value=evaluate_snapshot(model,data.load(version,next(model.parameters()).device),keeps,
                                partitions/f'{version}.pt',seen)
        atomic_json(done,value)
        atomic_json(output/'progress.json',dict(completed=i+1,total=len(data.groups),split=data.split))
    rows=[]
    sensitivity=[]
    for version in sorted(data.groups):
        value=json.loads((committed/f'{version}.json').read_text())
        rows.extend(value['rows']); sensitivity.extend(value['query_sensitivity'])
    from .metrics import distribution
    q_summary={f'keep={keep}/{name}':distribution([r['mean_tile_mask_jaccard'] for r in sensitivity
                 if r['keep']==keep and r['arm']==name]) for keep in keeps for name in ('A','B')}
    summary=dict(split=data.split,metrics=summarize(rows),query_sensitivity_groups=len(sensitivity),
                 query_sensitivity=q_summary,training=training_identity,evaluation_identity=signature,
                 selection='fixed update6000 paired checkpoint; Test not used for selection')
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'query_sensitivity.json',sensitivity)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--split',choices=['train','val','test'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one CUDA GPU')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=SnapshotData(args.data,args.split)
    model,value=load_pair(args.checkpoint,'cuda')
    if value['updates']!=6000 or value['identity']['source_sha256']!=source_identity():
        raise ValueError('Formal evaluation requires current-source paired update6000 checkpoint')
    model.eval()
    print(json.dumps(evaluate(model,data,args.output,checkpoint_sha=file_sha(args.checkpoint),
                              exposure=value['sampler'],training_identity=value['identity']['config']),indent=2))


if __name__=='__main__':
    main()
