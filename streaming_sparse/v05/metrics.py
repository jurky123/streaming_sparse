"""Energy-pooled attention errors alongside tile and independent-query metrics."""
from collections import defaultdict
import numpy as np


def distribution(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Nonfinite/empty metric')
    return dict(mean=float(values.mean()),median=float(np.median(values)),
                p90=float(np.quantile(values,.9)),p99=float(np.quantile(values,.99)))


def error(output,target):
    se=(output.float()-target.float()).square().sum(-1)
    en=target.float().square().sum(-1)
    return dict(sse=float(se.sum()),energy=float(en.sum()),error=float(se.sum()/(en.sum()+1e-8)),
                per_query_error=(se/(en+1e-8)).cpu().tolist())


def group_summary(rows):
    names=sorted(rows[0]['errors'])
    result={}
    for name in names:
        values=[row['errors'][name] for row in rows]
        result[name]=dict(global_attention_error=sum(v['sse'] for v in values)/(sum(v['energy'] for v in values)+1e-8),
                          tile_relative_error=distribution([v['error'] for v in values]),
                          query_relative_error=distribution([x for v in values for x in v['per_query_error']]))
    a=np.asarray([r['errors']['R2']['per_query_error'] for r in rows]).ravel()
    b=np.asarray([r['errors']['R3']['per_query_error'] for r in rows]).ravel()
    td=np.asarray([r['errors']['R2']['error']-r['errors']['R3']['error'] for r in rows])
    difference=result['R2']['global_attention_error']-result['R3']['global_attention_error']
    worst=sorted(rows,key=lambda r:r['errors']['R3']['error']-r['errors']['R2']['error'],reverse=True)[:10]
    result['paired']=dict(global_delta_R2_minus_R3=difference,
        relative_improvement=difference/max(result['R2']['global_attention_error'],1e-30),
        query_win_rate_R3=float(np.mean(b<a)),query_tie_rate=float(np.mean(b==a)),
        mean_query_improvement=float(np.mean(a-b)),median_query_improvement=float(np.median(a-b)),
        tile_win_rate_R3=float(np.mean(td>0)),mean_tile_improvement=float(td.mean()),
        worst_tile_regressions=[dict(cache_version=r['cache_version'],step_id=r['step_id'],tile_id=r['tile_id'],
                                    R3_minus_R2=r['errors']['R3']['error']-r['errors']['R2']['error']) for r in worst])
    result['tile_count']=len(rows);result['query_count']=len(a)
    result['retained_ratio']={name:distribution([x/r['history_tokens'] for r in rows for x in r['retained'][name]])
                              for name in rows[0]['retained']}
    result['budget_violations']=sum(x>r['budget'] for r in rows for v in r['retained'].values() for x in v)
    if result['budget_violations']:raise ValueError('Per-query budget violation')
    for field in ('correction_changed_mask_fraction_R2','correction_changed_mask_fraction_R3',
                  'positive_intervention_changed_mask_fraction','absolute_intervention_changed_mask_fraction'):
        if field in rows[0]:result[field]=distribution([r[field] for r in rows])
    return result


def summarize(rows):
    groups=defaultdict(list);seen=set()
    for row in rows:
        key=(row['cache_version'],row['step_id'],row['tile_id'],row['keep'])
        if key in seen:raise ValueError('Duplicate paired evaluation tile')
        seen.add(key)
        prefix=f"keep={row['keep']}"
        groups[prefix].append(row)
        for field in ('prompt_id','layer_id','head_id','history_tokens','seed','step_id','tile_id'):
            groups[f'{prefix}/{field}={row[field]}'].append(row)
        groups[f"{prefix}/layer_head={row['layer_id']}:{row['head_id']}"] .append(row)
    return {key:group_summary(value) for key,value in groups.items()}


def trace_summary(route,state):
    def stats(value):
        value=value.detach().float()
        return dict(mean=float(value.mean()),std=float(value.std(unbiased=False)),
                    mean_norm=float(value.norm(dim=-1).mean()))
    valid=~state['empty'];result={}
    for name,key in (('block_latent','h'),('aggregation','z_normalized')):
        result[name]=stats(state[key][valid])
    if 'features' in state:result['feature']=stats(state['features'])
    result['query_embedding']=stats(route['query_embedding'])
    result['fusion']=stats(route['fusion'][:,valid])
    result['relevance']=stats(route['relevance'][:,valid])
    result['correction']=stats(route['correction'][:,valid])
    layers=[]
    edge=valid[:,None]&valid[None,:]&~__import__('torch').eye(len(valid),device=valid.device,dtype=__import__('torch').bool)
    for layer in route['layers']:
        item=dict(residual_update_norm_ratio=float(layer['update'].detach().float().norm()/route['fusion'].detach().float().norm().clamp_min(1e-12)))
        if layer.get('relation') is not None:
            r=layer['relation'].detach()[:,edge].float()
            if r.numel():
                item.update(positive_fraction=float((r>1e-4).float().mean()),negative_fraction=float((r< -1e-4).float().mean()),
                            near_zero_fraction=float((r.abs()<=1e-4).float().mean()),abs_mean=float(r.abs().mean()),
                            abs_p90=float(__import__('torch').quantile(r.abs(),.9)),
                            abs_p99=float(__import__('torch').quantile(r.abs(),.99)))
            item['interaction_vector']=stats(layer['interaction_vector'][:,valid])
        layers.append(item)
    result['layers']=layers
    return result
