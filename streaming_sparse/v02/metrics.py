from collections import defaultdict
import random
import statistics
import torch
from scipy.optimize import linear_sum_assignment


def error_parts(output,target):
    return (output.float()-target.float()).square().sum(),target.float().square().sum()


def cluster_diagnostics(state):
    p=state['probabilities'].detach()
    hard=state['hard'].detach()
    labels=hard.argmax(-1)
    sizes=hard.sum(0)
    raw=state['raw_labels']
    raw_sizes=torch.bincount(raw,minlength=len(sizes)) if raw is not None else sizes
    return dict(soft_assignment_entropy=float(-(p*p.clamp_min(1e-12).log()).sum(-1).mean()),
        max_assignment_probability_mean=float(p.max(-1).values.mean()),
        max_assignment_probability_p10=float(p.max(-1).values.quantile(.1)),
        empty_before_repair=int((raw_sizes==0).sum()),empty_after_repair=int((sizes==0).sum()),
        repair_fraction=float((raw!=labels).float().mean()) if raw is not None else 0.,
        block_min=int(sizes.min()),block_max=int(sizes.max()),
        block_sizes=sizes.long().cpu().tolist(),blocks=len(sizes))


def partition_change(before,after):
    before,after=before.detach().long().cpu(),after.detach().long().cpu()
    b=max(int(before.max()),int(after.max()))+1
    contingency=torch.bincount(before*b+after,minlength=b*b).reshape(b,b)
    rows,cols=linear_sum_assignment(-contingency.numpy())
    matched=int(contingency[rows,cols].sum())
    return dict(raw_label_change=float((before!=after).float().mean()),
                permutation_aligned_change=1-matched/len(before))


def summarize(rows,bootstrap=1000):
    errors=torch.tensor([r['error'] for r in rows],dtype=torch.float64)
    prompts=defaultdict(list)
    for r in rows:
        prompts[r['prompt_id']].append(r)
    prompt_errors=[sum(x['squared_error'] for x in items)/(sum(x['dense_energy'] for x in items)+1e-8)
                   for items in prompts.values()]
    # Main multi-head result: normalize each physical (layer,head) over its
    # queries before combining heads, then aggregate independent prompts.
    # Keep the energy-weighted global ratio as a separate diagnostic.
    prompt_head_errors=[]
    for items in prompts.values():
        heads=defaultdict(list)
        for r in items: heads[(r['layer_id'],r['head_id'])].append(r)
        head_errors=[sum(r['squared_error'] for r in group)/(sum(r['dense_energy'] for r in group)+1e-8)
                     for group in heads.values()]
        prompt_head_errors.append(statistics.mean(head_errors))
    rng=random.Random(0)
    draws=sorted(statistics.mean(rng.choices(prompt_errors,k=len(prompt_errors))) for _ in range(bootstrap))
    head_draws=sorted(statistics.mean(rng.choices(prompt_head_errors,k=len(prompt_head_errors))) for _ in range(bootstrap))
    return dict(tiles=len(rows),prompts=len(prompts),
        global_relative_error=sum(x['squared_error'] for x in rows)/(sum(x['dense_energy'] for x in rows)+1e-8),
        macro_prompt_relative_error=statistics.mean(prompt_errors),
        macro_prompt_head_relative_error=statistics.mean(prompt_head_errors),
        prompt_head_ci95=[head_draws[int(.025*bootstrap)],head_draws[min(bootstrap-1,int(.975*bootstrap))]],
        prompt_ci95=[draws[int(.025*bootstrap)],draws[min(bootstrap-1,int(.975*bootstrap))]],
        prompt_variance=statistics.variance(prompt_errors) if len(prompt_errors)>1 else None,
        tile_mean=float(errors.mean()),tile_median=float(errors.median()),
        tile_p90=float(errors.quantile(.9)),tile_p95=float(errors.quantile(.95)),
        actual_keep=statistics.mean(r['actual_keep'] for r in rows))


def aggregate(rows,bootstrap=1000):
    groups=defaultdict(list)
    for r in rows:
        key=f"{r['method']}/keep={r['requested_keep']}"
        groups[key].append(r)
        for field in ('layer_id','head_id','step_id','history_tokens','prompt_id','seed'):
            groups[f'{key}/{field}={r[field]}'].append(r)
        groups[f"{key}/layer_head={r['layer_id']}:{r['head_id']}"] .append(r)
    return {k:summarize(values,bootstrap) for k,values in groups.items()}
