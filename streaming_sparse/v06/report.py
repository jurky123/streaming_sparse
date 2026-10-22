"""Validation-only progression decisions and three-seed development results."""
import argparse
import json
from pathlib import Path
import statistics
from .common import HERE, atomic_json, source_identity, protocol


def decide():
    p = protocol(); rule = p['stage3_gate']; cases = {k: [] for k in ('C1', 'C2', 'C3')}
    for seed in (0, 1, 2):
        run = HERE / f'runs/seed{seed}/stage2'
        result = json.loads((run / 'full_val/summary.json').read_text())
        if result['identity']['split'] != 'val' or result['identity']['source_sha256'] != source_identity():
            raise ValueError('Decision requires current-source full Val')
        if len(result['identity']['snapshots']) != 864:
            raise ValueError('Full864-snapshot Val required')
        values = result['metrics']['keep=0.2']; base = values['C0']['global_attention_error']
        curve = json.loads((run / 'validation_curve.json').read_text())
        end = json.loads((run / 'status.json').read_text())
        for kind in cases:
            score = values[kind]['global_attention_error']
            tail = [x for x in curve if 0 < x['updates'] <= end['arm_updates'][kind]][-3:]
            monitor_wins = sum(x['hard'][kind] < x['hard']['C0'] for x in tail)
            cases[kind].append(dict(seed=seed, C0=base, corrected=score, relative_gain=(base - score) / base,
                                    last3_monitor_wins=monitor_wins, correction_vs_C1=score - values['C1']['global_attention_error']))
    outcomes = {}; eligible = []
    for kind, seeds in cases.items():
        mean_gain = statistics.mean(x['relative_gain'] for x in seeds)
        full_wins = sum(x['relative_gain'] > 0 for x in seeds)
        stable_seeds = sum(x['last3_monitor_wins'] >= rule['require_last3_monitor_wins'] for x in seeds)
        passed = mean_gain >= rule['min_mean_relative_gain'] and min(full_wins, stable_seeds) >= rule['min_seeds_improved']
        if passed:
            eligible.append(kind)
        outcomes[kind] = dict(eligible=passed, mean_relative_gain=mean_gain, seeds_improved=full_wins,
                              stable_monitor_seeds=stable_seeds, seeds=seeds)
    decision = dict(source_sha256=source_identity(), selection_split='val', eligible=eligible, details=outcomes,
                    signed_retraining_required='C3' not in eligible, rule=rule,
                    no_test_reads=True, caveat=p['test_caveat'],
                    cross_block_value='Eligibility is vs C0; independent interaction value additionally requires C2/C3 better than C1 across seeds.')
    target = HERE / 'reports/stage2_decision.json'
    if target.exists() and json.loads(target.read_text()) != decision:
        raise ValueError('Cannot replace a different locked Validation decision')
    atomic_json(target, decision); print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


def final_report():
    decision = json.loads((HERE / 'reports/stage2_decision.json').read_text())
    output = dict(source_sha256=source_identity(), decision=decision, test_status=protocol()['test_caveat'], comparisons={})
    directories = [('frozen_core', 'stage2')]
    if decision['signed_retraining_required']:
        directories.append(('signed_retraining', 'signed'))
    for kind in decision['eligible']:
        directories.extend([(f'staged_{kind}', f'stage3_{kind}'), (f'scratch_{kind}', f'scratch_{kind}')])
    text = ['v0.6 final development review', '', protocol()['test_caveat'], '',
            'Each number below is pooled global query hard-mask error. Std is sample std across3 Router seeds.', '']
    for label, directory in directories:
        output['comparisons'][label] = {}
        for split in ('val', 'test'):
            summaries = [json.loads((HERE / f'runs/seed{s}/{directory}/full_{split}/summary.json').read_text()) for s in (0, 1, 2)]
            for keep in (.3, .2, .1):
                key = f'{split}/keep={keep}'; names = [n for n, v in summaries[0]['metrics'][f'keep={keep}'].items()
                                                    if isinstance(v, dict) and 'global_attention_error' in v and not n.endswith('_soft')]
                values = {}
                for name in names:
                    raw = [s['metrics'][f'keep={keep}'][name]['global_attention_error'] for s in summaries]
                    values[name] = dict(mean=statistics.mean(raw), sample_std=statistics.stdev(raw), seeds=raw)
                    text.append(f'{label} {key} {name}: {values[name]["mean"]:.8f} +/- {values[name]["sample_std"]:.8f}')
                output['comparisons'][label][key] = values
    output['offline_diagnostics'] = [json.loads((HERE / f'runs/seed{s}/offline_diagnostics/summary.json').read_text()) for s in (0, 1, 2)]
    atomic_json(HERE / 'reports/final_summary.json', output)
    (HERE / 'reports/RESULTS_SUMMARY.txt').write_text('\n'.join(text) + '\n')
    print('\n'.join(text), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--decide-stage3', action='store_true'); args = parser.parse_args()
    decide() if args.decide_stage3 else final_report()


if __name__ == '__main__':
    main()
