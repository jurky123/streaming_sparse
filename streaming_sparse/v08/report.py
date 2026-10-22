"""Aggregate all paired seeds and best/last/init; never select on Test."""
import argparse
import json
from pathlib import Path
from .common import HERE,protocol,source_identity,atomic_json
from .metrics import factorial


def run(split):
    p=protocol();result=dict(source_sha256=source_identity(),split=split,comparisons={})
    for tag in ('initial','best','last'):
        result['comparisons'][tag]={}
        for keep in (20,10,30):
            models={}
            for seed in p['training_seeds']:
                for group in p['groups']:
                    path=HERE/f'evaluations/{split}/seed{seed}/{group}/{tag}/keep{keep}/summary.json'
                    s=json.loads(path.read_text())
                    if s['source_sha256']!=source_identity():raise ValueError('Mixed experiment sources')
                    models[seed,group]=s
            result['comparisons'][tag][str(keep)]=factorial(models)
    atomic_json(HERE/f'reports/final_{split}.json',result)
    lines=[f'CAVIR v0.8 {split}：Prompt-equal network x0 NMSE','',
           'best由Val选择；initial/last均保留。10%/30%仅为20%训练后的预算泛化。']
    for tag,keeps in result['comparisons'].items():
        lines.append(f'\n{tag}, keep20%')
        for metric in ('G_F','G_J','I','J_V_vs_A_V'):
            x=keeps['20'][metric];lines.append(f"{metric}: {x['mean']:.8g}; CI={x['prompt_paired_95CI']}; per-seed={x['per_seed']}")
        lines.append(f"J-V relative gain={keeps['20']['joint_V_relative_gain']:.4%}; investment gate={keeps['20']['investment_gate']}")
    (HERE/f'reports/RESULTS_{split}.txt').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],required=True)
    run(p.parse_args().split)
