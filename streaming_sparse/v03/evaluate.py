"""B0..B5 evaluation with paired feature partitions and durable v0.2 journaling."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from dataclasses import asdict
import torch
from .model import load_router
from .data import SnapshotData
from .metrics import aggregate, decoder_diagnostics, error_parts, partition_diagnostics
from .paths import protect_output
from ..attention.block_statistics import differentiable_block_stats
from ..capture.dataset_io import atomic_save, query_slices
from ..evaluation.baselines import baseline_assignment, mass_selection, greedy_oracle
from ..v02.evaluation_journal import EvaluationJournal, identity, atomic_json


def statistics(q, sample, assignment):
    return differentiable_block_stats(q, sample['k_hist'], sample['v_hist'], assignment,
                                     sample['k_current'], sample['v_current'])


def source_identity():
    digest = hashlib.sha256()
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((Path(__file__).parents[1]/'router').glob('*.py'))
    paths += list((Path(__file__).parents[1]/'attention').glob('*.py'))
    paths += [Path(__file__).parents[1]/name for name in
              ('capture/dataset_io.py', 'evaluation/baselines.py', 'integration/self_forcing_adapter.py',
               'v02/data.py', 'v02/metrics.py', 'v02/evaluation_journal.py')]
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((Path(__file__).parent/'configs/protocol.json').read_bytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, data, output=None, keeps=(.3, .2, .1), baselines=True,
             versions=None, tiles_per_event=None, save_masks=False, bootstrap=200, oracle=False):
    model.eval()
    device = next(model.parameters()).device
    versions = sorted(data.groups) if versions is None else list(versions)
    if not versions or len(versions) != len(set(versions)) or any(v not in data.groups for v in versions):
        raise ValueError('Invalid snapshot evaluation subset')
    if not keeps or len(set(keeps)) != len(keeps) or any(not 0 <= k <= 1 for k in keeps):
        raise ValueError('Invalid evaluation budgets')
    if tiles_per_event is not None and tiles_per_event < 1:
        raise ValueError('tiles_per_event must be positive')
    learned = 'B4' if model.config.fusion == 'concat' else 'B5'
    methods = ['B3', learned] + (['B0', 'B1', 'B2'] if baselines else [])
    if oracle:
        methods.append('Greedy_diagnostic')
    expected = {}
    for version in versions:
        keys = set()
        for r in data.groups[version]:
            count = r['captured_queries']//model.config.query_tile
            count = min(count, tiles_per_event) if tiles_per_event else count
            keys.update((r['step_id'], i, method, keep) for i in range(count) for method in methods for keep in keeps)
        expected[version] = keys
    journal, records = None, []
    if output:
        output = protect_output(output)
        value = dict(base=identity(model, data, versions, keeps, baselines, 0, save_masks),
                     data=data.fingerprint(), source=source_identity(), methods=methods,
                     tiles_per_event=tiles_per_event, version='0.3')
        signature = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        journal = EvaluationJournal(Path(output), signature, versions, expected, set(), keeps)
        records = journal.records
    try:
        for index in range(journal.done if journal else 0, len(versions)):
            version = versions[index]
            events = data.load(version, device)
            first = events[0]
            state = model.precompute(first['k_hist'], first['v_hist'], first['token_positions'],
                                     first['head_id'], first['layer_id'])
            partition = partition_diagnostics(state)
            assignments = {'feature': state['hard']}
            if baselines:
                assignments.update({name: baseline_assignment(name, first['k_hist'], first['v_hist'],
                    first['token_positions'], model.config.target_size, model.config.max_blocks) for name in ('B0', 'B1', 'B2')})
            previous = {}
            if output and save_masks:
                atomic_save(dict(labels=state['hard'].argmax(-1).cpu(), diagnostics=partition,
                                 cache_version=version), Path(output)/'partitions'/f'{version}.pt')
            for event in events:
                slices = query_slices(event, model.config.query_tile)
                if tiles_per_event:
                    slices = slices[:tiles_per_event]
                for tile_index, selection in enumerate(slices):
                    q, target = event['q'][selection], event['o_dense'][selection]
                    stats = {name: statistics(q, event, a) for name, a in assignments.items()}
                    for keep in keeps:
                        route = model.route(q, state, keep)
                        budget = route['budget']
                        costs = state['hard'].sum(0)
                        masks = {'B3': mass_selection(stats['feature'], costs, budget), learned: route['hard_mask']}
                        if baselines:
                            masks.update({name: mass_selection(stats[name], assignments[name].sum(0), budget) for name in ('B0', 'B1', 'B2')})
                        if oracle:
                            masks['Greedy_diagnostic'] = greedy_oracle(stats['feature'], costs, budget, target)
                        for method, mask in masks.items():
                            source = method if method in assignments else 'feature'
                            block_costs = assignments[source].sum(0)
                            kept = int((block_costs*mask).sum())
                            if kept > budget:
                                raise AssertionError('History token budget exceeded')
                            se, energy = error_parts(stats[source].output(mask), target)
                            row = {k: event[k] for k in ('prompt_id', 'seed', 'layer_id', 'head_id', 'step_id', 'chunk_id', 'cache_version')}
                            row.update(method=method, partition_source=model.config.fusion if source == 'feature' else source,
                                requested_keep=keep, tile_index=tile_index, history_tokens=len(event['k_hist']),
                                query_indices=event['query_indices'][selection].cpu().tolist(),
                                kept_tokens=kept, token_budget=budget, actual_keep=kept/len(event['k_hist']),
                                budget_utilization=kept/max(budget, 1), squared_error=float(se),
                                dense_energy=float(energy), error=float(se/(energy+1e-8)))
                            if method == learned:
                                row['decoder'] = decoder_diagnostics(route, stats['feature'], target, state['hard'])
                                token_mask = state['hard']@mask > 0
                                key = (tile_index, keep)
                                query_ids = tuple(row['query_indices'])
                                if key in previous:
                                    last_step, last_queries, old = previous[key]
                                    if last_step == event['step_id']-1 and last_queries == query_ids:
                                        union = (token_mask | old).sum()
                                        row['step_mask_jaccard'] = float((token_mask & old).sum()/union) if int(union) else 1.
                                previous[key] = (event['step_id'], query_ids, token_mask.clone())
                                capped = mass_selection(stats['feature'], costs, kept)
                                capped_se, _ = error_parts(stats['feature'].output(capped), target)
                                row['mass_at_learned_token_cap_error'] = float(capped_se/(energy+1e-8))
                                if event['step_id'] == 0 and tile_index == 0 and keep == keeps[0]:
                                    row['partition'] = partition
                            if save_masks:
                                row['selected_blocks'] = torch.where(mask > 0)[0].cpu().tolist()
                            records.append(row)
                            if journal:
                                journal.raw.write(json.dumps(row)+'\n')
            if journal:
                # Same completeness checks used when reopening a durable journal.
                from ..v02.evaluation_journal import keys_match
                snapshot_rows = [r for r in records[-len(expected[version]):] if r['cache_version'] == version]
                if not keys_match(snapshot_rows, expected[version]):
                    raise AssertionError('Evaluation produced incomplete/duplicate rows')
                journal.commit(index+1)
    finally:
        if journal:
            journal.close()
    report = dict(version='0.3', snapshots=len(versions), fusion=model.config.fusion,
                  split=data.split, complete_data_protocol=data.require_complete,
                  heads=list(data.heads), layers=list(data.layers), source_sha256=source_identity(),
                  router_config=asdict(model.config), data_fingerprint=data.fingerprint(),
                  training_identity=getattr(model, 'run_identity', None),
                  primary='macro_prompt_head_relative_error', primary_keep=.2,
                  feature_mass_partition='Exactly the same learned checkpoint and hard assignment as B4/B5',
                  **aggregate(records, bootstrap))
    decoder_rows = [r['decoder'] for r in records if 'decoder' in r]
    partition_rows = [r['partition'] for r in records if 'partition' in r]
    def diagnostic_means(rows):
        if not rows:
            return {}
        result = {}
        for key in rows[0]:
            if isinstance(rows[0][key], (int, float)) or rows[0][key] is None:
                values = [float(row[key]) for row in rows if row[key] is not None]
                result[key] = sum(values)/len(values) if values else None
        return result
    changes = [r['step_mask_jaccard'] for r in records if 'step_mask_jaccard' in r]
    report['diagnostics'] = dict(decoder_mean=diagnostic_means(decoder_rows),
        partition_mean=diagnostic_means(partition_rows), step_mask_jaccard_mean=sum(changes)/len(changes) if changes else None)
    if output:
        atomic_json(Path(output)/'summary.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    p.add_argument('--heads', type=int, nargs='+', default=[0, 3, 6, 9])
    p.add_argument('--device', default='cuda')
    p.add_argument('--oracle', action='store_true', help='Slow evaluation-only greedy diagnostic; never training supervision')
    p.add_argument('--save-masks', action='store_true')
    args = p.parse_args()
    data = SnapshotData(args.data, args.split, heads=args.heads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not start a real evaluation')
    torch.backends.cuda.matmul.allow_tf32 = False
    report = evaluate(load_router(args.checkpoint, args.device), data, args.output,
                      save_masks=args.save_masks, oracle=args.oracle)
    print(json.dumps(report['comparisons'], indent=2))


if __name__ == '__main__':
    main()
