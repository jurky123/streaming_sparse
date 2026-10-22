"""Pooled errors, query/tile tails and paired mask-change outcomes."""
from collections import defaultdict
from itertools import combinations
import numpy as np
from ..v05.metrics import distribution, error, trace_summary


def group_summary(rows):
    names = sorted(rows[0]['errors'])
    result = {}
    for name in names:
        values = [r['errors'][name] for r in rows]
        result[name] = dict(global_attention_error=sum(v['sse'] for v in values) / (sum(v['energy'] for v in values) + 1e-8),
                            tile_relative_error=distribution([v['error'] for v in values]),
                            query_relative_error=distribution([x for v in values for x in v['per_query_error']]))
    result['paired'] = {}
    models = [n for n in names if n.startswith('C') and not n.endswith('_soft')]
    for a, b in combinations(models, 2):
        aa = np.asarray([r['errors'][a]['per_query_error'] for r in rows]).ravel()
        bb = np.asarray([r['errors'][b]['per_query_error'] for r in rows]).ravel()
        delta = aa - bb
        worst = np.argsort(delta)[:10]
        result['paired'][f'{b}_vs_{a}'] = dict(query_win_rate=float(np.mean(delta > 0)),
            query_tie_rate=float(np.mean(delta == 0)), mean_query_improvement=float(delta.mean()),
            median_query_improvement=float(np.median(delta)),
            global_improvement=result[a]['global_attention_error'] - result[b]['global_attention_error'],
            worst_query_regressions=[dict(cache_version=rows[i // 16]['cache_version'],
                step_id=rows[i // 16]['step_id'], tile_id=rows[i // 16]['tile_id'],
                query_in_tile=int(i % 16), delta=float(delta[i])) for i in worst])
    result['correction_effect'] = {}
    for name in rows[0].get('mask_changes', {}):
        changed = np.asarray([r['mask_changes'][name] for r in rows], dtype=bool).ravel()
        base = np.asarray([r['errors']['C0']['per_query_error'] for r in rows]).ravel()
        value = np.asarray([r['errors'][name]['per_query_error'] for r in rows]).ravel()
        d = (base - value)[changed]
        result['correction_effect'][name] = dict(changed_fraction=float(changed.mean()), changed_queries=int(changed.sum()),
            improved_given_change=float(np.mean(d > 0)) if len(d) else None,
            worsened_given_change=float(np.mean(d < 0)) if len(d) else None,
            tied_given_change=float(np.mean(d == 0)) if len(d) else None,
            mean_improvement_given_change=float(d.mean()) if len(d) else None)
    result['tile_count'] = len(rows); result['query_count'] = 16 * len(rows)
    result['retained_ratio'] = {name: distribution([x / r['history_tokens'] for r in rows for x in r['retained'][name]])
                                for name in rows[0]['retained']}
    result['budget_violations'] = sum(x > r['budget'] for r in rows for values in r['retained'].values() for x in values)
    if result['budget_violations']:
        raise ValueError('Per-query budget violation')
    return result


def summarize(rows):
    groups = defaultdict(list); seen = set()
    for r in rows:
        key = (r['cache_version'], r['step_id'], r['tile_id'], r['keep'])
        if key in seen:
            raise ValueError('Duplicate evaluation tile')
        seen.add(key); prefix = f"keep={r['keep']}"
        groups[prefix].append(r)
        for field in ('prompt_id', 'layer_id', 'head_id', 'history_tokens', 'seed', 'step_id', 'tile_id'):
            groups[f'{prefix}/{field}={r[field]}'].append(r)
        groups[f"{prefix}/layer_head={r['layer_id']}:{r['head_id']}"] .append(r)
    return {key: group_summary(value) for key, value in groups.items()}
