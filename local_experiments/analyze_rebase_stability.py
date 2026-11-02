"""Descriptive temporal-stability proxies for existing paired videos; not a quality metric."""
import argparse
import csv
import json
from pathlib import Path

import av
import numpy as np
from scipy.ndimage import median_filter

from .run_kv_history import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    args = parser.parse_args()
    root = Path(args.experiment)
    out = root / 'stability_analysis'
    out.mkdir(exist_ok=False)
    assert json.loads((root / 'result.json').read_text())['completed']
    with (root / 'attention_mass_by_chunk.csv').open() as f:
        chunks = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    chunks = [x for x in chunks if x['chunk_1_based'] >= 7]
    videos = {}
    for variant in ('baseline', 'rebased'):
        with av.open(str(root / f'{variant}.mp4')) as reader:
            fps = float(reader.streams.video[0].average_rate)
            mean_rgb = np.stack([frame.to_ndarray(format='rgb24').mean(axis=(0, 1)) / 255
                                 for frame in reader.decode(video=0)])
        assert len(mean_rgb) == 480 and fps == 16
        brightness = mean_rgb @ np.array([0.2126, 0.7152, 0.0722])
        chromaticity = mean_rgb / mean_rgb.sum(axis=1, keepdims=True)
        videos[variant] = dict(mean_rgb=mean_rgb, brightness=brightness, chromaticity=chromaticity,
                               dark_dip=np.maximum(0, median_filter(brightness, size=9, mode='nearest') - brightness))
    # Shared pre-intervention video prefix supplies a single reference, never a fitted per-variant one.
    color_reference = videos['baseline']['chromaticity'][:69].mean(axis=0)
    summary = {}
    for variant, data in videos.items():
        y, c = data['brightness'], data['chromaticity']
        data['color_drift'] = np.linalg.norm(c - color_reference, axis=1)
        attention = np.array([x[f'{variant}_mass'] for x in chunks])
        # Both endpoints must be after frame 69, where the generated trajectories first diverge.
        dy = np.diff(y[69:])
        d2y = np.diff(y[69:], n=2)
        color_steps = np.linalg.norm(np.diff(c[69:], axis=0), axis=1)
        dips = data['dark_dip'][69:]
        summary[variant] = dict(
            attention_chunk_step_mean_abs_pp=float(np.abs(np.diff(attention)).mean() * 100),
            attention_temporal_std_pp=float(attention.std() * 100),
            brightness_frame_step_mean_abs_pp=float(np.abs(dy).mean() * 100),
            brightness_frame_step_p95_abs_pp=float(np.quantile(np.abs(dy), .95) * 100),
            brightness_frame_step_max_abs_pp=float(np.abs(dy).max() * 100),
            brightness_second_difference_rms_pp=float(np.sqrt(np.mean(d2y**2)) * 100),
            dark_dip_p95_pp=float(np.quantile(dips, .95) * 100),
            dark_dip_max_pp=float(dips.max() * 100),
            dark_dip_frames_over_2pp=int((dips > .02).sum()),
            chromaticity_frame_step_mean=float(color_steps.mean()),
            chromaticity_frame_step_p95=float(np.quantile(color_steps, .95)),
            chromaticity_drift_from_prefix_mean=float(data['color_drift'][69:].mean()))
    reductions = {k: 100 * (1 - summary['rebased'][k] / summary['baseline'][k])
                  for k in summary['baseline'] if summary['baseline'][k] > 0}
    for variant, data in videos.items():
        with (out / f'{variant}.frames.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_zero_based', 'seconds', 'mean_R', 'mean_G', 'mean_B',
                             'brightness_proxy', 'nine_frame_median_dark_dip', 'chromaticity_r',
                             'chromaticity_g', 'chromaticity_b', 'color_drift_from_common_prefix'])
            for i in range(480):
                writer.writerow([i, i / 16, *data['mean_rgb'][i], data['brightness'][i],
                                 data['dark_dip'][i], *data['chromaticity'][i], data['color_drift'][i]])
    definitions = dict(
        interval='Video frames 69..479 (4.3125..29.9375 s); attention chunks 7..41. Differences use endpoints inside the interval.',
        brightness='Spatial mean of 0.2126 R + 0.7152 G + 0.0722 B, with decoded nonlinear RGB code values normalized to [0,1]. A display-brightness proxy, not physical luminance.',
        dark_dip='max(0, centered 9-frame median brightness - brightness). Nearest padding at video ends. Fixed 2-percentage-point threshold; descriptive, not perceptually calibrated.',
        chromaticity='Spatial mean RGB normalized by R+G+B. Drift is Euclidean distance from mean baseline chromaticity in shared frames 0..68.',
        limitations='Camera movement, scene composition and intentional lighting changes affect all global video proxies; compression is included. These are not motion-compensated/object-tracked color or flicker scores.',
        replication='One prompt, one seed. Temporal standard deviation describes fluctuations, not uncertainty over independent trials.')
    write_json(out / 'summary.json', dict(metrics=summary, reduction_percent=reductions, definitions=definitions))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
        time = np.arange(480) / 16
        for variant, color, style, label in [('baseline', '#0072B2', '-', 'Original time'),
                                             ('rebased', '#D55E00', '--', 'Rebased time')]:
            data = videos[variant]
            axes[0, 0].plot([x['video_start_second'] for x in chunks],
                            [100 * x[f'{variant}_mass'] for x in chunks],
                            color=color, linestyle=style, marker='o' if variant == 'baseline' else 's',
                            markersize=3, label=label)
            axes[0, 1].plot(time, 100 * data['brightness'], color=color, linestyle=style, label=label)
            axes[1, 0].plot(time, 100 * data['dark_dip'], color=color, linestyle=style, label=label)
            axes[1, 1].plot(time, data['color_drift'], color=color, linestyle=style, label=label)
        titles = ['Anchor attention after intervention', 'Global display-brightness proxy',
                  'Short dark dips relative to 9-frame median', 'Global chromaticity distance from common prefix']
        labels = ['Attention mass (%)', 'Brightness proxy (%)', 'Dark dip (percentage points)', 'RGB chromaticity distance']
        for ax, title, ylabel in zip(axes.flat, titles, labels):
            ax.set(title=title, xlabel='Video time (s)', ylabel=ylabel, xlim=(0, 30))
            ax.axvline(69 / 16, color='#666666', linestyle=':', linewidth=1)
            ax.legend(fontsize=8)
            ax.grid(axis='y', color='#dddddd', linewidth=.6)
        axes[0, 0].set_xlim(4, 30)
        for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
            ax.set_ylim(bottom=0)
        fig.suptitle('Temporal stability: descriptive proxies, one prompt / seed 0', fontsize=14)
        fig.supxlabel('Global brightness/color changes also reflect motion, composition and lighting. Dotted line: intervention begins. No curve smoothing.', fontsize=8)
        fig.savefig(out / 'stability.png', dpi=180, facecolor='white')
        fig.savefig(out / 'stability.svg', facecolor='white')
        plt.close(fig)
    write_json(out / 'provenance.json', dict(
        sources={str(root / n): sha(root / n) for n in ('baseline.mp4', 'rebased.mp4', 'attention_mass_by_chunk.csv')},
        analysis_source_sha256=sha(__file__), definitions=definitions, matplotlib=matplotlib.__version__))
    (out / 'description.txt').write_text(
        'Four panels show anchor attention, global brightness, short dark dips and RGB chromaticity drift. '
        'Blue solid lines indicate original positions; orange dashed lines indicate rebased positions. '
        'All observations are shown. The dotted line marks intervention at 4.3125 seconds. '
        'Per-frame CSVs and summary.json provide the data and definitions; no quality or causal-mechanism proof is implied.\n')
    print(json.dumps(dict(metrics=summary, reduction_percent=reductions), indent=2))


if __name__ == '__main__':
    main()
