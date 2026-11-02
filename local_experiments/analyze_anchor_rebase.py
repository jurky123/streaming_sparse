"""Summarize unsmoothed sampled attention mass; no quality or significance inference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .run_kv_history import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder')
    a = p.parse_args()
    folder = Path(a.folder)
    assert json.loads((folder / 'result.json').read_text())['completed']
    manifest = json.loads((folder / 'manifest.json').read_text())
    by_variant = {}
    for variant in ('baseline', 'rebased'):
        rows = [json.loads(x) for x in (folder / f'{variant}.attention.jsonl').read_text().splitlines()]
        assert len(rows) == 40 * 5 * 30
        assert all(r['query_count'] == 48 and len(r['mass']) == 12 for r in rows)
        assert all(np.isfinite(r['mass']).all() and min(r['mass']) >= 0 and max(r['mass']) <= 1 for r in rows)
        by_variant[variant] = rows
    points = []
    for chunk in range(1, 41):
        point = dict(chunk_1_based=chunk + 1, video_start_second=(12 * chunk - 3) / 16,
                     rebased_anchor_first_position=max(0, 3 * chunk - 15))
        for variant, rows in by_variant.items():
            selected = [r for r in rows if r['chunk'] == chunk and r['kind'] == 'denoise']
            assert {(r['step'], r['layer']) for r in selected} == {(s, l) for s in range(4) for l in range(30)}
            point[f'{variant}_mass'] = float(np.mean([r['mass'] for r in selected]))
            if variant == 'baseline':
                point['baseline_state_rebased_mass'] = float(np.mean([r['same_state_rebased_mass'] for r in selected]))
        points.append(point)
    with (folder / 'attention_mass_by_chunk.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(points[0]))
        w.writeheader()
        w.writerows(points)
    windows = {}
    for name, selected in (
        ('post_intervention', [p for p in points if p['chunk_1_based'] >= 7]),
        ('last_10_seconds', [p for p in points if p['video_start_second'] >= 20])):
        averages = {key: float(np.mean([r[key] for r in selected])) for key in
                    ('baseline_mass', 'rebased_mass', 'baseline_state_rebased_mass')}
        averages.update(chunks=len(selected),
                        actual_change_percentage_points=100 * (averages['rebased_mass'] - averages['baseline_mass']),
                        phase_only_change_percentage_points=100 * (averages['baseline_state_rebased_mass'] - averages['baseline_mass']))
        windows[name] = averages
    layer_rows = []
    for layer in range(30):
        row = dict(layer_zero_based=layer)
        for variant, records in by_variant.items():
            selected = [r for r in records if r['layer'] == layer and r['kind'] == 'denoise' and (12 * r['chunk'] - 3) / 16 >= 20]
            row[f'{variant}_mass'] = float(np.mean([r['mass'] for r in selected]))
            if variant == 'baseline':
                row['baseline_state_rebased_mass'] = float(np.mean([r['same_state_rebased_mass'] for r in selected]))
        layer_rows.append(row)
    with (folder / 'attention_mass_last10s_by_layer.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
        w.writeheader()
        w.writerows(layer_rows)
    summary = dict(windows=windows, independent_prompts=1, independent_seeds=1,
                   query_sampling='Fixed 4x4 spatial grid per each of 3 query latent frames: 48 queries, all 12 heads and 30 layers.',
                   aggregation='Arithmetic mean of per-head sampled-query mean mass, equally across 12 heads, 30 layers and 4 denoise steps. Window summaries average chunks equally, including the final padded chunk.',
                   exclusions='Chunk 0 has no historical anchor; clean-context forwards excluded from plots but preserved in raw JSONL.',
                   uncertainty='No confidence intervals: one prompt/seed; layers, heads and chunks are not independent replicate videos.',
                   interpretation='Actual rollout changes include recurrent content feedback. Same-state counterfactual on baseline Q/K isolates the immediate temporal-phase effect. Attention mass does not establish video quality.')
    write_json(folder / 'attention_summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [p['video_start_second'] for p in points]
    baseline = np.array([p['baseline_mass'] for p in points]) * 100
    treatment = np.array([p['rebased_mass'] for p in points]) * 100
    counterfactual = np.array([p['baseline_state_rebased_mass'] for p in points]) * 100
    ymax = min(100, max(baseline.max(), treatment.max(), counterfactual.max()) * 1.15)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharey=True, layout='constrained')
        for ax, series, title, label in zip(axes, [treatment, counterfactual],
                ['Actual generation trajectories', 'Same baseline Q/K: phase-only effect'],
                ['Dynamic anchor time', 'Rebased anchor, same content']):
            ax.plot(x, baseline, color='#0072B2', marker='o', markersize=3, linewidth=1.5, label='Original anchor time')
            ax.plot(x, series, color='#D55E00', marker='s', markersize=3, linestyle='--', linewidth=1.5, label=label)
            ax.axvline(69 / 16, color='#666666', linestyle=':', linewidth=1)
            ax.set(title=title, xlabel='Query chunk start in output video (s)', xlim=(0, 30), ylim=(0, ymax))
            ax.grid(axis='y', color='#dddddd', linewidth=0.6)
            ax.legend(fontsize=8, loc='best')
        axes[0].set_ylabel('Mean attention mass on first chunk (%)')
        fig.suptitle('Anchor attention: one prompt, seed 0', fontsize=14)
        fig.supxlabel('48 sampled queries; all layers/heads; mean over 4 denoise steps. Dotted line: intervention begins. No smoothing.', fontsize=8)
        fig.savefig(folder / 'attention_mass.png', dpi=180, facecolor='white')
        fig.savefig(folder / 'attention_mass.svg', facecolor='white')
        plt.close(fig)
    description = (
        'Two panels compare first-chunk attention mass versus video time. Left: actual original and rebased rollouts. '
        'Right: original and rebased anchor phases evaluated on identical baseline Q/K. '
        'Blue solid circles are original positions; orange dashed squares are rebased positions. '
        'All observations are shown without smoothing. Chunk 0 is omitted because it has no historical anchor. '
        'One prompt and seed; no statistical or video-quality conclusion is implied. Data: attention_mass_by_chunk.csv.\n')
    (folder / 'attention_mass_description.txt').write_text(description)
    write_json(folder / 'figure_provenance.json', dict(
        matplotlib=matplotlib.__version__, source_files={name: sha(folder / name) for name in
        ('baseline.attention.jsonl', 'rebased.attention.jsonl', 'attention_mass_by_chunk.csv')},
        analysis_source_sha256=sha(__file__), output_pixels=[2250, 846], dpi=180,
        transformations=summary['aggregation'], exclusions=summary['exclusions'],
        uncertainty=summary['uncertainty'], destination='Exploratory local experiment; no journal-specific formatting.'))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
