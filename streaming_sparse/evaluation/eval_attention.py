import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import torch
from ..capture.dataset_io import AttentionDataset, query_slices
from ..router.model import load_router
from ..router.budget_selection import select_under_budget
from ..attention.block_statistics import differentiable_block_stats
from ..attention.reconstruction_loss import normalized_attention_mse
from ..train.train_router import to_device
from .baselines import baseline_assignment, mass_selection, greedy_oracle


def summary(records):
    errors = torch.tensor([x['error'] for x in records], dtype=torch.float64)
    by_prompt = defaultdict(list)
    for r in records:
        by_prompt[r['prompt_id']].append(r['error'])
    means = [statistics.mean(x) for x in by_prompt.values()]
    rng = random.Random(0)
    bootstrap = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(1000))
    result = dict(count=len(records), prompts=len(means), mean=float(errors.mean()), median=float(errors.quantile(.5)),
                p90=float(errors.quantile(.9)), prompt_variance=statistics.variance(means) if len(means)>1 else None,
                prompt_mean_ci95=[bootstrap[25], bootstrap[974]],
                mean_actual_keep=statistics.mean(r['actual_keep'] for r in records))
    result['generation_seeds'] = sorted({r['seed'] for r in records if r.get('seed') is not None})
    result['prompt_seed_pairs'] = len({(r['prompt_id'],r.get('seed')) for r in records})
    if 'hard_soft_gap' in records[0]:
        result['mean_hard_soft_gap'] = statistics.mean(r['hard_soft_gap'] for r in records)
    if 'delta_error_vs_ours' in records[0]:
        matched = [r for r in records if r['token_difference_vs_ours'] == 0]
        result.update(mean_delta_error_vs_ours=statistics.mean(r['delta_error_vs_ours'] for r in records),
                      exact_token_match_count=len(matched),
                      exact_token_matched_delta=statistics.mean(r['delta_error_vs_ours'] for r in matched) if matched else None,
                      mean_token_difference_vs_ours=statistics.mean(r['token_difference_vs_ours'] for r in records))
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--b3-checkpoint')
    p.add_argument('--split', choices=['train','val','test'], default='test')
    p.add_argument('--keep', nargs='+', type=float, default=[0.3, 0.2, 0.1])
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    p.add_argument('--max-events', type=int)
    p.add_argument('--oracle-max-blocks', type=int, default=0)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    router = load_router(args.checkpoint, args.device).eval()
    b3 = load_router(args.b3_checkpoint, args.device).eval() if args.b3_checkpoint else None
    # Loading optional models must not change random-selection control masks.
    torch.manual_seed(0)
    if b3 and b3.config.grouping != 'spatial':
        raise ValueError('B3 requires a separately trained spatial grouping router')
    dataset = AttentionDataset(args.data, args.split)
    if not len(dataset):
        raise ValueError('Empty evaluation split')
    # Sort snapshots so H is reused across steps while bounding device memory.
    indices = sorted(range(len(dataset)), key=lambda i: dataset.records[i]['cache_version'])
    if args.max_events:
        indices = indices[:args.max_events]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, previous, states, assignments = [], None, {}, {}
    with (output / 'per_tile.jsonl').open('w') as f:
        for index in indices:
            sample = to_device(dataset[index], args.device)
            if previous != sample['cache_version']:
                previous = sample['cache_version']
                states = {'Ours': router.precompute(sample['k_hist'], sample['v_hist'], sample['token_positions'],
                                                  sample['head_id'], sample['layer_id'])}
                if b3:
                    states['B3'] = b3.precompute(sample['k_hist'], sample['v_hist'], sample['token_positions'],
                                                sample['head_id'], sample['layer_id'])
                assignments = {name: baseline_assignment(name, sample['k_hist'], sample['v_hist'], sample['token_positions'],
                        router.config.target_size, router.config.max_blocks) for name in ('B0','B1','B2')}
                assignments.update({name: s['hard'] for name, s in states.items()})
            tile = router.config.query_tile
            for selection in query_slices(sample, tile):
                start = selection.start
                q = sample['q'][selection]
                target = sample['o_dense'][selection]
                stats = {name: differentiable_block_stats(q, sample['k_hist'], sample['v_hist'], a,
                          sample['k_current'], sample['v_current']) for name, a in assignments.items()}
                for keep in args.keep:
                    if not 0 <= keep <= 1:
                        raise ValueError('Invalid keep ratio')
                    budget = math.floor(keep * len(sample['k_hist']))
                    ours = router.route(q, states['Ours'], keep)
                    ours_kept = int((assignments['Ours'].sum(0) * ours['hard_mask']).sum())
                    routes = {name: mass_selection(stats[name], a.sum(0), budget) for name, a in assignments.items() if name.startswith('B') and name != 'B3'}
                    routes['Ours'] = ours['hard_mask']
                    if b3:
                        routes['B3'] = b3.route(q, states['B3'], keep)['hard_mask']
                    # A paired mass comparator on the SAME learned partition and
                    # capped at Ours' actual token count, with utilization reported.
                    routes['Mass_same_partition'] = mass_selection(stats['Ours'], assignments['Ours'].sum(0), ours_kept)
                    routes['Random'] = select_under_budget(
                        torch.rand(assignments['Ours'].shape[1], device=q.device, generator=None), assignments['Ours'].sum(0), budget)
                    if 0 < len(ours['hard_mask']) <= args.oracle_max_blocks:
                        routes['Greedy_reference'] = greedy_oracle(stats['Ours'], assignments['Ours'].sum(0), budget, target)
                    ours_error = float(normalized_attention_mse(stats['Ours'].output(ours['hard_mask']), target))
                    for name, mask in routes.items():
                        source = name if name in assignments else 'Ours'
                        costs = assignments[source].sum(0)
                        kept = int((costs * mask).sum())
                        record = {key: sample[key] for key in ('prompt_id','layer_id','head_id','chunk_id','step_id','cache_version')}
                        record['seed'] = sample.get('seed')
                        record.update(method=name, requested_keep=keep, actual_keep=kept/len(sample['k_hist']),
                            kept_tokens=kept, history_tokens=len(sample['k_hist']), tile_start=start, blocks=len(costs),
                            error=float(normalized_attention_mse(stats[source].output(mask), target)))
                        if name == 'Ours':
                            record['soft_error'] = float(normalized_attention_mse(stats['Ours'].output(ours['soft_mask']), target))
                            record['hard_soft_gap'] = record['error'] - record['soft_error']
                        if name == 'Mass_same_partition':
                            record['delta_error_vs_ours'] = record['error'] - ours_error
                            record['token_difference_vs_ours'] = kept - ours_kept
                        records.append(record)
                        f.write(json.dumps(record) + '\n')
    groups = defaultdict(list)
    for r in records:
        groups[f"{r['method']}/keep={r['requested_keep']}"] .append(r)
        for field in ('layer_id','head_id','step_id','history_tokens','seed','prompt_id'):
            groups[f"{r['method']}/keep={r['requested_keep']}/{field}={r[field]}"] .append(r)
        groups[f"{r['method']}/keep={r['requested_keep']}/prompt_seed={r['prompt_id']}:{r['seed']}"] .append(r)
    report = {'metrics': {key: summary(values) for key, values in groups.items()},
              'b3_evaluated': b3 is not None, 'split': args.split,
              'note': 'Compare actual token utilization before interpreting delta; greedy reference is not a proven upper bound.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report['metrics'].items() if k.count('/') == 1}, indent=2))


if __name__ == '__main__':
    main()
