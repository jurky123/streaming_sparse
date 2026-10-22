"""Aggregate the complete final-checkpoint matrix without changing model selection."""
import argparse
import json
from pathlib import Path
from .metrics import across_router_seeds
from .identity import source_identity
from ..v02.evaluation_journal import atomic_json
from .paths import protect_output


def run(runs,output):
    runs=Path(runs)
    collected={}
    for keep in (.3,.2):
        for split in ('train','val','test'):
            seeds={}
            for seed in (0,1,2):
                path=runs/f'keep{int(keep*100)}/seed{seed}/evaluation_{split}/summary.json'
                if not path.is_file():raise FileNotFoundError(f'Incomplete matrix: {path}')
                value=json.loads(path.read_text())
                config=value['training']
                if config['train_keep']!=keep or config['router_seed']!=seed or value['split']!=split:
                    raise ValueError('Mislabelled seed/budget/split result')
                if value['evaluation_identity']['source_sha256']!=source_identity():
                    raise ValueError('Result from a different source version')
                seeds[seed]=value['metrics']
            collected[f'train_keep={keep}/{split}']=across_router_seeds(seeds)
    output=protect_output(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,dict(version='0.4',source_sha256=source_identity(),router_seed_std_ddof=1,
        generation_seeds='Both0/1 within every router run; sliced separately, not extra independent replicates',
        summaries=collected,interpretation='Assess unseen prompts, heads/history/steps/tile consistency; do not infer usefulness from one mean'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run(args.runs,args.output)


if __name__=='__main__':main()
