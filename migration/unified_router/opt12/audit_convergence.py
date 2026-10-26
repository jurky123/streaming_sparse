"""Audit raw validation convergence and persist per-layer evidence."""
import json
import statistics
from pathlib import Path
from ..common import atomic_json
from .status import ROOT


def main():
    root = ROOT
    records, summary = [], {}
    for variant in ['Compact', 'Aggressive']:
        rows = []
        for layer in range(1, 30):
            folder = root / f'training/layer{layer:02d}'
            done = json.loads((folder / 'completed.json').read_text())
            curve = json.loads((folder / variant / 'curve.json').read_text())
            best = json.loads((folder / variant / 'best_validation.json').read_text())
            end, prev = curve[-1], curve[-2]
            earlier = next(x for x in curve if x['updates'] == end['updates'] - 500)
            train = {}
            for line in (folder / variant / 'train.jsonl').read_text().splitlines():
                r = json.loads(line)
                train[r['updates']] = r
            recent = [train[i] for i in sorted(train) if i > end['updates'] - 500]
            row = dict(layer=layer, variant=variant, end_step=done['updates'], reason=done['reason'],
                best_step=best['updates'], best_nmse=best['nmse'], final_nmse=end['nmse'],
                initial_nmse=curve[0]['nmse'], final_keep=end['keep'],
                last100_improvement_pct=100*(1-end['nmse']/prev['nmse']),
                last500_improvement_pct=100*(1-end['nmse']/earlier['nmse']),
                best_at_final=best['updates'] == end['updates'],
                last500_train_actual_nmse=statistics.mean(r['actual_nmse'] for r in recent),
                last500_revisions=sorted(set(r.get('revision', 'original_v1') for r in recent)), curve=curve)
            row['still_improving_at_cap'] = (row['end_step'] == 2000 and row['best_at_final'] and
                row['last100_improvement_pct'] > 0 and row['last500_improvement_pct'] > 0)
            rows.append(row)
        summary[variant] = dict(initial_nmse=statistics.mean(r['initial_nmse'] for r in rows),
            best_nmse=statistics.mean(r['best_nmse'] for r in rows),
            capped_layers=[r['layer'] for r in rows if r['end_step'] == 2000],
            still_improving_at_cap=[r['layer'] for r in rows if r['still_improving_at_cap']],
            earlier_best_layers=[r['layer'] for r in rows if r['best_step'] < r['end_step']])
        records.extend(rows)
    atomic_json(root / 'convergence_audit.json', dict(summary=summary, rows=records,
        criterion='at2000, best at2000, and raw actual-validation NMSE improves over both last100 and last500 updates',
        caveat='evidence of remaining improvement, not proof that longer training improves video quality; gradient-precision revisions recorded'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for layer, ax in zip([21, 25, 26, 27, 28, 29], axes.flat):
        for variant in ['Compact', 'Aggressive']:
            row = next(r for r in records if r['layer'] == layer and r['variant'] == variant)
            curve = row['curve']
            anchor = next(x['nmse'] for x in curve if x['updates'] == 1500)
            points = [x for x in curve if x['updates'] >= 1500]
            ax.plot([x['updates'] for x in points], [(x['nmse']/anchor-1)*100 for x in points], '-o', label=variant)
        ax.set_title(f'Layer {layer}: raw validation, last 500 updates')
        ax.set_xlabel('Updates')
        ax.set_ylabel('NMSE change vs step1500 (%)')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Opt12: continued improvement at the 2000-update cap (lower is better)')
    fig.savefig(root/'convergence_tail.png', dpi=150)
    fig.savefig(root/'convergence_tail.pdf')
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
