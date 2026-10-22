"""Offline-only fixed-query greedy headroom and exact numerator interactions."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from .common import (DEFAULT_DATA, SnapshotData, fixed_diagnostic_samples, atomic_json,
                     source_identity, protect_output, file_sha, query_slices)
from .evaluate import compose
from ..v05.primitives import statistics, mass_mask


def numerator_vectors(stat, dense):
    return stat.u - stat.z[..., None] * dense[:, None]


def omitted_identity(stat, dense, mask):
    a = numerator_vectors(stat, dense)
    missing = 1 - mask.float()
    numerator = (a * missing[..., None]).sum(1)
    retained_z = stat.z_current + (stat.z * mask).sum(-1)
    return -numerator / retained_z[:, None].clamp_min(1e-30)


@torch.no_grad()
def approx_greedy(stat, target, costs, budget):
    if len(target) != 1:
        raise ValueError('Fixed diagnostic greedy is one raw query at a time')
    mask = torch.zeros_like(stat.z); remaining = int(budget)
    numerator = stat.u_current[0].clone(); denominator = stat.z_current[0].clone()
    evaluations = 0; order = []
    current_error = (numerator / denominator.clamp_min(1e-30) - target[0]).square().sum()
    for _ in range(len(costs)):
        candidates = (costs > 0) & (costs <= remaining) & (mask[0] == 0)
        if not candidates.any():
            break
        outputs = (numerator[None] + stat.u[0]) / (denominator + stat.z[0]).clamp_min(1e-30)[:, None]
        errors = (outputs - target[0]).square().sum(-1).masked_fill(~candidates, torch.inf)
        evaluations += int(candidates.sum())
        index = int(errors.argmin())
        if not errors[index] < current_error:
            break
        current_error = errors[index]; mask[0, index] = 1; order.append(index)
        remaining -= int(costs[index]); numerator += stat.u[0, index]; denominator += stat.z[0, index]
    return mask, dict(order=order, retained=int(budget - remaining), candidate_evaluations=evaluations,
                      stopping='No affordable improving single-block addition; approximate local search, not an oracle optimum')


def correlation(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    nonzero = (np.abs(a) > 1e-12) & (np.abs(b) > 1e-12)
    pearson = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-15 else None
    spearman = float(spearmanr(a, b).statistic) if min(a.std(), b.std()) > 1e-15 else None
    return dict(edges=len(a), pearson=pearson, spearman=spearman,
                nonzero_sign_agreement=float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero]))) if nonzero.any() else None)


@torch.no_grad()
def run(stage2, output):
    model, checkpoint_hashes = compose(stage2, 'cuda')
    data = SnapshotData(DEFAULT_DATA, 'val'); device = 'cuda'
    output = protect_output(output); output.mkdir(parents=True, exist_ok=True)
    plan = fixed_diagnostic_samples(data)
    identity = dict(source_sha256=source_identity(), checkpoint_hashes=checkpoint_hashes,
                    data_fingerprint=data.fingerprint(), samples=plan, split='val', keep=.2,
                    no_training_labels=True, role='Offline diagnosis only, not checkpoint or structure selection')
    if (output / 'identity.json').exists() and json.loads((output / 'identity.json').read_text()) != identity:
        raise ValueError('Offline diagnostic identity mismatch')
    atomic_json(output / 'identity.json', identity); reports = []
    for i, sample in enumerate(plan):
        events = data.load(sample['cache_version'], device); first = events[0]
        state = model.precompute(first['k_hist'], first['v_hist'], head_id=first['head_id'], layer_id=first['layer_id'])
        event = next(e for e in events if e['step_id'] == sample['step_id'])
        tile = query_slices(event, 16)[sample['tile_id']]; index = tile.start + sample['query_in_tile']
        q = event['q'][index:index + 1]; target = event['o_dense'][index:index + 1].float()
        stat = statistics(q, event, state['assignment']); dense = stat.dense()
        torch.testing.assert_close(dense, target, atol=5e-5, rtol=5e-5)
        routes = model.route(q, state, .2, trace=True); costs = state['hard'].sum(0); budget = math.floor(.2 * len(state['hard']))
        masks = {name: route['hard_mask'] for name, route in routes.items()}
        masks['Mass'] = mass_mask(stat, costs, budget, state['block_ids'])
        masks['ApproxGreedy'], greedy = approx_greedy(stat, target, costs, budget)
        # Use exact statistics-dense output for the algebraic identity; capture target
        # can differ by a small documented backend rounding error.
        identity_errors = []
        for mask in masks.values():
            reconstructed = omitted_identity(stat, dense, mask)
            torch.testing.assert_close(reconstructed, stat.output(mask) - dense, atol=3e-5, rtol=5e-4)
            identity_errors.append(float((reconstructed - (stat.output(mask) - dense)).abs().max()))
        a = numerator_vectors(stat, dense)[0]
        gram = a @ a.T
        cosine = torch.nn.functional.normalize(a, dim=-1) @ torch.nn.functional.normalize(a, dim=-1).T
        valid = ~state['empty']; edges = torch.triu(valid[:, None] & valid[None, :], diagonal=1)
        relation_stats = []; arrays = dict(true_numerator_gram=gram.cpu().numpy(), true_numerator_cosine=cosine.cpu().numpy(),
                                          costs=costs.cpu().numpy(), block_ids=state['block_ids'].cpu().numpy())
        for j, layer in enumerate(routes['C3']['layers']):
            r = layer['relation'][0]; values = r[edges].float().cpu().numpy()
            relation_stats.append(dict(layer=j, vs_gram=correlation(values, gram[edges].cpu().numpy()),
                                      vs_cosine=correlation(values, cosine[edges].cpu().numpy()),
                                      positive_fraction=float(np.mean(values > 1e-4)),
                                      negative_fraction=float(np.mean(values < -1e-4))))
            arrays[f'learned_relation{j}'] = r.cpu().numpy()
        errors = {}
        for name, mask in masks.items():
            se = float((stat.output(mask) - target).square().sum()); energy = float(target.square().sum())
            errors[name] = dict(sse=se, energy=energy, relative_error=se / (energy + 1e-8),
                                retained=int((mask * costs).sum()), selected_blocks=mask[0].nonzero().flatten().cpu().tolist())
        asset = output / f'query{i:02d}.npz'; np.savez_compressed(asset, **arrays)
        reports.append(dict(**sample, layer=first['layer_id'], head=first['head_id'], partition_hash=state['partition_hash'],
                            query_index=int(event['query_indices'][index]), budget=budget, errors=errors, greedy=greedy,
                            relations=relation_stats, max_identity_error=max(identity_errors),
                            raw_asset=asset.name, raw_sha256=file_sha(asset)))
        atomic_json(output / 'progress.json', dict(completed=i + 1, total=len(plan)))
    pooled = {name: sum(r['errors'][name]['sse'] for r in reports) / (sum(r['errors'][name]['energy'] for r in reports) + 1e-8)
              for name in reports[0]['errors']}
    atomic_json(output / 'summary.json', dict(identity=identity, queries=reports, pooled_errors=pooled,
        warning='Small metadata-fixed Val subset; greedy can underfill, is approximate, and need not dominate Mass. Numerator interaction is not the full normalized mask error.'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage2', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one GPU')
    run(args.stage2, args.output)


if __name__ == '__main__':
    main()
