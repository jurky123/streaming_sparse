"""Query-micro and prompt/video-group macro metrics; paired group bootstrap."""
from collections import defaultdict
from itertools import combinations
import numpy as np
import torch


def distribution(values):
    x=np.asarray(values,dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Empty/nonfinite metric')
    return dict(mean=float(x.mean()),median=float(np.median(x)),p25=float(np.quantile(x,.25)),p75=float(np.quantile(x,.75)),
                p90=float(np.quantile(x,.9)),p99=float(np.quantile(x,.99)))


def error(output,target):
    se=(output.float()-target.float()).square().sum(-1);en=target.float().square().sum(-1)
    if not torch.isfinite(se).all() or not torch.isfinite(en).all():raise FloatingPointError('Nonfinite exact attention error')
    return dict(sse=float(se.sum()),energy=float(en.sum()),query_sse=se.cpu().tolist(),query_energy=en.cpu().tolist(),
                per_query_error=(se/(en+1e-8)).cpu().tolist(),error=float(se.sum()/(en.sum()+1e-8)))


def group_summary(rows,bootstrap=False):
    methods=sorted(rows[0]['errors']);result={};groups=defaultdict(list)
    for r in rows:groups[(r['prompt_id'],r['seed'])].append(r)
    group_values={}
    for name in methods:
        gv={str(g):sum(r['errors'][name]['sse'] for r in rs)/(sum(r['errors'][name]['energy'] for r in rs)+1e-8) for g,rs in groups.items()}
        group_values[name]=gv
        result[name]=dict(micro_error=sum(r['errors'][name]['sse'] for r in rows)/(sum(r['errors'][name]['energy'] for r in rows)+1e-8),
            macro_error=float(np.mean(list(gv.values()))),group_distribution=distribution(list(gv.values())),groups=gv,
            query_error=distribution([x for r in rows for x in r['errors'][name]['per_query_error']]),
            tile_error=distribution([r['errors'][name]['error'] for r in rows]))
    result['paired']={}
    for a,b in combinations([m for m in methods if not m.endswith('_soft')],2):
        ga=np.asarray(list(group_values[a].values()));gb=np.asarray(list(group_values[b].values()))
        aa=np.asarray([r['errors'][a]['per_query_error'] for r in rows]).ravel();bb=np.asarray([r['errors'][b]['per_query_error'] for r in rows]).ravel()
        diff=ga-gb;delta=aa-bb
        item=dict(macro_improvement=float(diff.mean()),relative_macro_improvement=float(diff.mean()/max(ga.mean(),1e-30)),
                  group_win_fraction=float(np.mean(diff>0)),query_win_fraction=float(np.mean(delta>0)),query_tie_fraction=float(np.mean(delta==0)),
                  query_improvement_mean=float(delta.mean()),query_improvement_median=float(np.median(delta)))
        if bootstrap:
            rng=np.random.default_rng(20261018);sample=rng.integers(len(diff),size=(2000,len(diff)))
            boot=diff[sample].mean(-1);item['paired_group_bootstrap_95CI']=np.quantile(boot,[.025,.975]).tolist()
            item['bootstrap_units']='(prompt_id,generation seed) groups; queries are not independent units'
        result['paired'][f'{b}_vs_{a}']=item
    result['correction']={}
    for name in rows[0].get('changed',{}):
        changed=np.asarray([r['changed'][name] for r in rows],dtype=bool).ravel()
        baseline='R3' if 'R3' in methods else 'T2'
        dd=np.asarray([r['errors'][baseline]['per_query_error'] for r in rows]).ravel()-np.asarray([r['errors'][name]['per_query_error'] for r in rows]).ravel()
        result['correction'][name]=dict(changed_fraction=float(changed.mean()),improved_given_change=float(np.mean(dd[changed]>0)) if changed.any() else None,
            worsened_given_change=float(np.mean(dd[changed]<0)) if changed.any() else None,
            jaccard=distribution([x for r in rows for x in r['jaccard'][name]]))
    result['retained_ratio']={m:distribution([x/r['history_tokens'] for r in rows for x in r['retained'][m]]) for m in rows[0]['retained']}
    if 'sketch_mask_overlap' in rows[0]:
        result['sketch_true_singleton_mask_jaccard']=distribution([x for r in rows for x in r['sketch_mask_overlap']])
    result['budget_violations']=sum(x>r['budget'] for r in rows for vs in r['retained'].values() for x in vs)
    if result['budget_violations']:raise AssertionError('History token budget violation')
    result.update(tile_count=len(rows),query_count=sum(len(r['query_indices']) for r in rows),video_group_count=len(groups))
    if bootstrap and 'interaction' in rows[0]:
        kappa=np.asarray([x for r in rows for x in r['interaction']['kappa']])
        boundaries=[-np.inf,-.9,-.5,0,.5,np.inf];bins={}
        for lo,hi in zip(boundaries,boundaries[1:]):
            selected=(kappa>=lo)&(kappa<hi)
            if not selected.any():continue
            values={}
            for name in methods:
                se=np.asarray([x for r in rows for x in r['errors'][name]['query_sse']])
                en=np.asarray([x for r in rows for x in r['errors'][name]['query_energy']])
                values[name]=float(se[selected].sum()/(en[selected].sum()+1e-8))
            bins[f'[{lo},{hi})']=dict(query_count=int(selected.sum()),micro_error=values)
        result['true_interaction_bins']=bins
    return result


def summarize(rows):
    groups=defaultdict(list);seen=set()
    for r in rows:
        key=(r['cache_version'],r['step_id'],r['tile_id'],r['keep'])
        if key in seen:raise ValueError('Duplicated paired event')
        seen.add(key);prefix=f"keep={r['keep']}";groups[prefix].append(r)
        for field in ('prompt_id','seed','layer_id','head_id','step_id','history_tokens'):
            groups[f'{prefix}/{field}={r[field]}'].append(r)
        groups[f"{prefix}/pair={r['layer_id']}:{r['head_id']}"] .append(r)
        groups[f"{prefix}/history_step={r['history_tokens']}:{r['step_id']}"] .append(r)
    return {k:group_summary(v,bootstrap='/' not in k) for k,v in groups.items()}


def sketch_losses(pred,true):
    g=true.geometry()
    z=(pred['logz']-true.logz).square().mean()
    v=(g['mass']*(pred['mu']-true.mu).square().sum(-1)).mean()
    target=true.dense();o=((pred['dense']-target).square().sum(-1)/(target.square().sum(-1)+1e-8)).mean()
    return dict(Z=z,V=v,O=o)


@torch.no_grad()
def sketch_metrics(pred,true):
    g=true.geometry();z=pred['logz']-true.logz;vdiff=(pred['mu']-true.mu).square().sum(-1)
    dn=g['d'].norm(dim=-1);pn=pred['d'].norm(dim=-1);valid=(dn>1e-8)&(pn>1e-8)
    cosine=(pred['d']*g['d']).sum(-1)/(pn*dn).clamp_min(1e-30)
    return dict(logz_mae=float(z.abs().mean()),logz_mse=float(z.square().mean()),
                weighted_logz_abs=float((z.abs()*g['mass']).sum(-1).mean()),
                logz_count=z.numel(),logz_abs_sum=float(z.abs().sum()),logz_sq_sum=float(z.square().sum()),
                logz_mass_sum=float(g['mass'].sum()),logz_mass_abs_sum=float((z.abs()*g['mass']).sum()),
                logz_mass_sq_sum=float((z.square()*g['mass']).sum()),
                mass_l1=float((pred['mass']-g['mass']).abs().sum(-1).mean()),
                current_mass_abs=float((pred['current_mass']-g['current_mass']).abs().mean()),
                weighted_value_error=float((vdiff*g['mass']).sum(-1).mean()),
                high_mass_value_error=float((vdiff*(g['mass']>.01)).sum()/((g['mass']>.01).sum().clamp_min(1))),
                residual_cosine_sum=float(cosine[valid].sum()),residual_valid_count=int(valid.sum()),
                residual_low_norm_count=int((~valid).sum()),residual_norm_abs_bias=float((pn-dn).abs().mean()),
                dense_error=error(pred['dense'],true.dense()))
