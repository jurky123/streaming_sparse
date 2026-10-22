"""Summarize real 4-head evidence before permitting supplemental 12-head capture."""
import argparse
import json
import math
from pathlib import Path
from .evaluate import source_identity
from .paths import protect_output
from ..v02.evaluation_journal import atomic_json


def architecture_gate(summaries, training_runs, overfit_reports):
    reasons, comparisons = [], {}
    source = source_identity()
    if len(summaries) != 2 or len(training_runs) != 2:
        raise ValueError('Provide one summary and training run for each fusion')
    for summary_path, run_path in zip(summaries, training_runs):
        summary = json.loads(Path(summary_path).read_text())
        fusion = summary['fusion']
        method = 'B4' if fusion == 'concat' else 'B5'
        if fusion in comparisons:
            raise ValueError('Both distinct fusions must be checked')
        if (summary['split'] != 'val' or not summary['complete_data_protocol'] or
            set(summary['heads']) != {0, 3, 6, 9} or summary['snapshots'] != 288 or
            summary['source_sha256'] != source):
            reasons.append(f'{fusion}: require complete, current-source 4-head validation, never test')
        metric = summary['metrics'][f'{method}/keep=0.2']['macro_prompt_head_relative_error']
        mass = summary['metrics']['B3/keep=0.2']['macro_prompt_head_relative_error']
        comparisons[fusion] = dict(learned_error=metric, same_partition_mass_error=mass,
                                  ratio=metric/mass if mass else None)
        if not math.isfinite(metric) or not math.isfinite(mass) or metric > mass:
            reasons.append(f'{fusion}: learned routing has not reached same-partition Mass at 20%')
        run = Path(run_path)
        config = json.loads((run/'config.json').read_text())
        rows = [json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
        if config['router']['fusion'] != fusion or config['stage'] != 'pilot' or len(rows) != 1500:
            reasons.append(f'{fusion}: require the completed 1500-update implementation pilot')
        prov = json.loads((run/'provenance.json').read_text())
        if prov['source_sha256'] != source or prov['validation_data_fingerprint'] != summary['data_fingerprint']:
            reasons.append(f'{fusion}: training/validation evidence identity mismatch')
        if summary.get('training_identity') != prov:
            reasons.append(f'{fusion}: validation did not use the specified training run')
        gradients = rows[0].get('first_update_gradient_gate')
        if not gradients or any(not math.isfinite(v) or v <= 0 for v in gradients.values()):
            reasons.append(f'{fusion}: first-update gradient gate missing/failed')
        if any(not math.isfinite(r['attention_loss']) for r in rows):
            reasons.append(f'{fusion}: nonfinite training loss')
        for row in rows:
            for diagnostic in row['diagnostics']:
                part = diagnostic['partition']
                if part['empty_after_repair'] or not math.isfinite(part['feature_norm_mean']) or part['feature_norm_mean'] <= 1e-8:
                    reasons.append(f'{fusion}: invalid/collapsed feature partition')
                    break
    if set(comparisons) != {'concat', 'film'}:
        reasons.append('Missing a fusion')
    if not any(v['learned_error'] < v['same_partition_mass_error'] for v in comparisons.values()):
        reasons.append('No fusion improves on same-partition Mass at 20%')
    overfits = [json.loads(Path(p).read_text()) for p in overfit_reports]
    if {x.get('fusion') for x in overfits if x.get('passed') and x.get('real_qkv') and x.get('source_sha256') == source} != {'concat', 'film'}:
        reasons.append('Both fusions must pass real-QKV single-snapshot overfit')
    return dict(version='0.3', source_sha256=source, capture_allowed=not reasons,
                reasons=sorted(set(reasons)), comparisons=comparisons,
                operational_rule='Both fusions no worse than same-partition Mass at 20%; at least one strictly better. No significance claim.',
                summaries=[str(p) for p in summaries], training_runs=[str(p) for p in training_runs],
                overfit_reports=[str(p) for p in overfit_reports])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summaries', nargs=2, required=True, help='Full pilot validation summary.json, concat then film')
    p.add_argument('--training-runs', nargs=2, required=True)
    p.add_argument('--overfit-reports', nargs=2, required=True)
    p.add_argument('--output', default='streaming_sparse/v03/reports/architecture_gate.json')
    args = p.parse_args()
    target = protect_output(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = architecture_gate(args.summaries, args.training_runs, args.overfit_reports)
    atomic_json(target, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
