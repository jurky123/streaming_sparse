from collections import defaultdict
import math
import statistics as py_stats
import torch
from ..v02.metrics import error_parts, summarize as summarize_v02


def partition_diagnostics(state):
    hard, p = state['hard'].detach(), state['probabilities'].detach()
    sizes = hard.sum(0)
    raw = torch.bincount(state['raw_labels'], minlength=len(sizes))
    norms = state['features'].detach().norm(dim=-1)
    return dict(blocks=len(sizes), block_min=float(sizes.min()), block_median=float(sizes.median()),
                block_p90=float(sizes.quantile(.9)), block_max=float(sizes.max()),
                block_sizes=sizes.int().cpu().tolist(), empty_before_repair=int((raw == 0).sum()),
                empty_after_repair=int((sizes == 0).sum()),
                repair_fraction=float((state['raw_labels'] != hard.argmax(-1)).float().mean()),
                assignment_entropy=float(-(p*p.clamp_min(1e-12).log()).sum(-1).mean()),
                feature_norm_mean=float(norms.mean()), feature_norm_std=float(norms.std(unbiased=False)))


def correlation(a, b):
    a, b = a.detach().float(), b.detach().float()
    a, b = a-a.mean(), b-b.mean()
    denominator = a.norm()*b.norm()
    return float((a*b).sum()/denominator) if float(denominator) > 1e-12 else None


def decoder_diagnostics(route, stats, target, hard):
    se, energy = error_parts(stats.output(route['hard_mask']), target)
    soft_se, _ = error_parts(stats.output(route['soft_mask']), target)
    costs = hard.sum(0)
    kept = int((route['hard_mask']*costs).sum())
    mass = (stats.z/(stats.z.sum(-1)+stats.z_current)[:, None].clamp_min(1e-30)).sum(0)
    scores = route['scores'].detach()
    return dict(score_mean=float(scores.mean()), score_std=float(scores.std(unbiased=False)),
                score_mass_correlation=correlation(scores, mass),
                hard_error=float(se/(energy+1e-8)), soft_error=float(soft_se/(energy+1e-8)),
                hard_soft_gap=float((se-soft_se)/(energy+1e-8)), actual_keep=kept/len(hard),
                budget_utilization=kept/max(route['budget'], 1), kept_tokens=kept, token_budget=route['budget'])


def mask_change(masks):
    result = []
    for first, second in zip(masks, masks[1:]):
        union = (first | second).sum()
        result.append(float((first & second).sum()/union) if int(union) else 1.)
    return result


def gradient_gate(model, state):
    modules = dict(joint_feature=model.features, transform=model.aggregator.transform,
                   block_projection=model.aggregator.projection, pre_q=model.encoder,
                   q_projection=model.decoder.query, q_fusion=model.decoder.fusion,
                   post_q=model.decoder.post, score=model.decoder.score)
    norms = {name: math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters()
                                 if p.grad is not None)) for name, module in modules.items()}
    norms['cluster_assignment'] = float(state['probabilities'].grad.norm()) if state['probabilities'].grad is not None else 0.
    norms['cluster_feature'] = float(state['features'].grad.norm()) if state['features'].grad is not None else 0.
    norms['cluster_feature_via_assignment'] = state.get('cluster_feature_path_norm', 0.)
    if model.config.conditioned:
        norms['active_pair_residual'] = math.sqrt(sum(float(p.grad[state['pair']].square().sum())
            for name, p in model.named_parameters() if 'delta_' in name and p.grad is not None))
    if any(not math.isfinite(v) or v <= 0 for v in norms.values()):
        raise AssertionError(f'First-update gradient gate failed: {norms}')
    return norms


def summarize(rows, bootstrap=200):
    result = summarize_v02(rows, bootstrap=bootstrap)
    result['prompt_energy_weighted_error'] = result['macro_prompt_relative_error']
    result['prompt_energy_weighted_definition'] = 'Within each prompt pool SSE/energy across heads, then average prompts equally'
    result['budget_utilization'] = py_stats.mean(r['budget_utilization'] for r in rows)
    return result


def aggregate(rows, bootstrap=200):
    groups = defaultdict(list)
    for r in rows:
        key = f"{r['method']}/keep={r['requested_keep']}"
        groups[key].append(r)
        for field in ('layer_id', 'head_id', 'step_id', 'history_tokens', 'prompt_id', 'seed'):
            groups[f'{key}/{field}={r[field]}'].append(r)
        groups[f"{key}/layer_head={r['layer_id']}:{r['head_id']}"] .append(r)
    metrics = {k: summarize(v, bootstrap) for k, v in groups.items()}
    comparisons = {}
    for key, value in metrics.items():
        if key.startswith(('B4/', 'B5/')):
            mass_key = 'B3/' + key.split('/', 1)[1]
            if mass_key in metrics:
                mass = metrics[mass_key]['macro_prompt_head_relative_error']
                learned = value['macro_prompt_head_relative_error']
                comparisons[key] = dict(learned_over_same_partition_mass=learned/mass if mass else None,
                                        learned_error=learned, mass_error=mass)
    return dict(metrics=metrics, comparisons=comparisons)
