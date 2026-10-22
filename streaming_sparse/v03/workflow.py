"""Print an experiment command matrix. This module never starts a job."""
import argparse
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]


def plan(stage, data, output, shared=False):
    jobs = []
    seeds = (0,) if stage == 'pilot' else (0, 1, 2)
    heads = [0, 3, 6, 9] if stage == 'pilot' else list(range(12))
    for fusion in ('concat', 'film'):
        config = f'streaming_sparse/v03/configs/{"fusion" if stage == "pilot" else "full"}_{fusion}.json'
        for seed in seeds:
            target = str(Path(output)/stage/(fusion+('_shared' if shared else ''))/f'seed{seed}')
            train = ['python', '-m', 'streaming_sparse.v03.train', '--config', config,
                     '--data', str(data), '--output', target, '--seed', str(seed)]
            if shared:
                train.append('--shared')
            jobs.append(dict(kind='train', stage=stage, fusion=fusion, seed=seed, argv=train))
            # Pilot architecture decisions use validation only. Full final evaluation
            # includes test only after best.pt has been selected on validation.
            for split in (('val',) if stage == 'pilot' else ('val', 'test')):
                jobs.append(dict(kind='evaluate', split=split, fusion=fusion, seed=seed,
                    argv=['python', '-m', 'streaming_sparse.v03.evaluate', '--data', str(data),
                          '--checkpoint', target+'/best.pt', '--split', split,
                          '--output', target+f'/evaluation_{split}', '--heads', *map(str, heads)]))
    return dict(version='0.3', automatic_execution=False, stage=stage, shared=shared, jobs=jobs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['pilot', 'full'], required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', default='streaming_sparse/v03/runs')
    p.add_argument('--shared', action='store_true')
    args = p.parse_args()
    result = plan(args.stage, args.data, args.output, args.shared)
    for job in result['jobs']:
        print(shlex.join(job['argv']))


if __name__ == '__main__':
    main()
