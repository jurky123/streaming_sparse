"""Aggregate paired learned/Mass results over router seeds, preserving partitions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from .paths import protect_output
from ..v02.evaluation_journal import atomic_json


def analyze(paths, require_full_seeds=False):
    groups = defaultdict(list)
    for path in paths:
        value = json.loads(Path(path).read_text())
        if not value['complete_data_protocol']:
            raise ValueError('Synthetic/development results cannot become a research report')
        identity = value.get('training_identity')
        if not identity:
            raise ValueError('Research reports require training/checkpoint provenance')
        key = (value['fusion'], value['router_config']['conditioned'], value['split'], tuple(value['heads']))
        groups[key].append((identity['config']['seed'], value, str(path)))
    report = {}
    for (fusion, conditioned, split, heads), rows in groups.items():
        seeds = [seed for seed, _, _ in rows]
        if len(set(seeds)) != len(seeds) or (require_full_seeds and set(seeds) != {0, 1, 2}):
            raise ValueError('Require distinct router seeds; Full requires exactly 0,1,2')
        if require_full_seeds and set(heads) != set(range(12)):
            raise ValueError('Formal Full reports require all twelve heads')
        if len({r['data_fingerprint'] for _, r, _ in rows}) != 1 or len({r['source_sha256'] for _, r, _ in rows}) != 1:
            raise ValueError('Seed comparisons require identical data and source')
        method = 'B4' if fusion == 'concat' else 'B5'
        budgets = {}
        for keep in (.3, .2, .1):
            metrics = {}
            for name in ('B0', 'B1', 'B2', 'B3', method):
                errors = [r['metrics'][f'{name}/keep={keep}']['macro_prompt_head_relative_error'] for _, r, _ in rows]
                metrics[name] = dict(mean=statistics.mean(errors), seed_std=statistics.stdev(errors) if len(errors) > 1 else None)
            ratios = [r['comparisons'][f'{method}/keep={keep}']['learned_over_same_partition_mass'] for _, r, _ in rows]
            ratios = [r for r in ratios if r is not None]
            budgets[str(keep)] = dict(metrics=metrics, learned_over_mass_mean=statistics.mean(ratios) if ratios else None,
                                     learned_over_mass_seed_std=statistics.stdev(ratios) if len(ratios) > 1 else None)
        strong = all(budgets[str(k)]['metrics'][method]['mean'] < budgets[str(k)]['metrics']['B3']['mean'] < budgets[str(k)]['metrics']['B0']['mean'] for k in (.2, .1))
        useful = budgets['0.2']['metrics'][method]['mean'] < budgets['0.2']['metrics']['B3']['mean']
        report[f'{fusion}/conditioned={conditioned}/{split}/heads={len(heads)}'] = dict(
            seeds=sorted(seeds), budgets=budgets, strong_success_at_20_and_10=strong,
            useful_success_at_20=useful, sources=[path for _, _, path in rows],
            note='B3 is always paired with the corresponding checkpoint; seed variability is not a significance test.')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--require-full-seeds', action='store_true')
    args = p.parse_args()
    report = analyze(args.reports, args.require_full_seeds)
    output = protect_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
