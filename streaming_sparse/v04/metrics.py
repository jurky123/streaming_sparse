"""Energy-global primary metric; query-tile tails and genuinely paired comparisons."""
from collections import defaultdict
import math
import numpy as np


def distribution(values):
    x=np.asarray(values,dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Missing or nonfinite metric values')
    return dict(mean=float(x.mean()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),
                p99=float(np.quantile(x,.99)))


def group_summary(rows):
    if not rows:
        raise ValueError('Empty paired sample group')
    result={}
    for name in ('A','B','Mass'):
        values=[r['errors'][name] for r in rows]
        result[name]=dict(global_attention_error=sum(v['sse'] for v in values)/(sum(v['energy'] for v in values)+1e-8),
                          tile_relative_error=distribution([v['error'] for v in values]))
    delta=[r['errors']['B']['error']-r['errors']['A']['error'] for r in rows]
    difference=result['B']['global_attention_error']-result['A']['global_attention_error']
    result['paired']=dict(global_delta_B_minus_A=difference,
        global_relative_improvement=difference/result['B']['global_attention_error'] if result['B']['global_attention_error']>0 else None,
        mean_improvement=float(np.mean(delta)),median_improvement=float(np.median(delta)),
        win_rate_A=float(np.mean(np.asarray(delta)>0)),tie_rate=float(np.mean(np.asarray(delta)==0)),
        worst_regressions=[dict(cache_version=r['cache_version'],step_id=r['step_id'],tile_id=r['tile_id'],
                               A_minus_B=r['errors']['A']['error']-r['errors']['B']['error'])
                           for r in sorted(rows,key=lambda r:r['errors']['A']['error']-r['errors']['B']['error'],reverse=True)[:10]])
    result['sample_count']=len(rows)
    result['budget_violations']=sum(r['retained'][name]>r['budget'] for r in rows for name in ('A','B','Mass'))
    if result['budget_violations']:
        raise ValueError('Budget violation in evaluation rows')
    return result


def summarize(rows):
    groups=defaultdict(list)
    seen=set()
    for r in rows:
        identity=(r['cache_version'],r['step_id'],r['tile_id'],r['keep'])
        if identity in seen:
            raise ValueError(f'Duplicate paired sample: {identity}')
        seen.add(identity)
        if r['partition_hash_A']!=r['partition_hash_B']:
            raise ValueError('A/B partitions differ')
        key=f"keep={r['keep']}"
        groups[key].append(r)
        for field in ('prompt_id','layer_id','head_id','history_tokens','seed','step_id','tile_id',
                      'query_tile_seen_in_training'):
            groups[f'{key}/{field}={r[field]}'].append(r)
        groups[f"{key}/layer_head={r['layer_id']}:{r['head_id']}"] .append(r)
    result={key:group_summary(values) for key,values in groups.items()}
    for keep in sorted({r['keep'] for r in rows}):
        head={name:float(np.var([value[name]['global_attention_error'] for key,value in result.items()
                    if key.startswith(f'keep={keep}/head_id=')],ddof=0)) for name in ('A','B','Mass')}
        result[f'keep={keep}']['head_error_variance']=head
        values=groups[f'keep={keep}']
        delta=np.asarray([r['errors']['B']['error']-r['errors']['A']['error'] for r in values])
        diagnostics={}
        for field in ('query_entropy_all_raw_KV','history_block_mass_concentration',
                      'z_mean_pair_cosine_redundancy_proxy','A_B_token_jaccard'):
            if any(field not in r for r in values):continue
            x=np.asarray([r[field] for r in values])
            a,b=x-x.mean(),delta-delta.mean()
            denom=float(np.linalg.norm(a)*np.linalg.norm(b))
            edges=np.quantile(x,[0,.25,.5,.75,1])
            bins=[]
            for i in range(4):
                chosen=(x>=edges[i])&((x<edges[i+1]) if i<3 else (x<=edges[i+1]))
                bins.append(dict(lower=float(edges[i]),upper=float(edges[i+1]),count=int(chosen.sum()),
                    mean_delta=float(delta[chosen].mean()) if chosen.any() else None))
            diagnostics[field]=dict(pearson_with_B_minus_A=float(a@b/denom) if denom>0 else None,
                                     quartiles=bins,interpretation='Descriptive association, not causal attribution')
        result[f'keep={keep}']['interaction_diagnostics']=diagnostics
    return result


def across_router_seeds(summaries):
    if set(summaries)!={0,1,2}:
        raise ValueError('Report requires router seeds0/1/2, independent of generation seeds')
    reference=set(summaries[0])
    if any(set(v)!=reference for v in summaries.values()):
        raise ValueError('Router-seed reports have different slices')
    output={}
    for key in sorted(reference):
        output[key]={}
        for name in ('A','B','Mass'):
            x=[summaries[s][key][name]['global_attention_error'] for s in (0,1,2)]
            output[key][name]=dict(mean=float(np.mean(x)),std=float(np.std(x,ddof=1)),router_seed_values=x)
            for tail in ('mean','median','p90','p99'):
                values=[summaries[s][key][name]['tile_relative_error'][tail] for s in (0,1,2)]
                output[key][name]['tile_'+tail]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
        for field in ('global_delta_B_minus_A','global_relative_improvement','win_rate_A',
                      'mean_improvement','median_improvement'):
            x=[summaries[s][key]['paired'][field] for s in (0,1,2)]
            x=[v for v in x if v is not None]
            output[key][field]=dict(mean=float(np.mean(x)) if x else None,
                                    std=float(np.std(x,ddof=1)) if len(x)>1 else None,valid_seeds=len(x))
    return output
