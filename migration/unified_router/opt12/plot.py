"""Plot recorded train loss and raw validation separately, never replace with best."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .status import ROOT


def plot(root=ROOT):
    destination = root / 'curves'
    destination.mkdir(exist_ok=True)
    count = 0
    for layer in range(1, 30):
        folder = root / f'training/layer{layer:02d}'
        if not list(folder.glob('*/train.jsonl')):
            continue
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for v, color in [('Compact', 'tab:blue'), ('Aggressive', 'tab:orange')]:
            path = folder / v / 'train.jsonl'
            records = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records[r['updates']] = r
            rows = [records[i] for i in sorted(records)]
            if rows:
                x = [r['updates'] for r in rows]
                y = np.array([r['actual_nmse'] for r in rows])
                axs[0, 0].plot(x, y, color=color, alpha=.15)
                window = min(50, len(y))
                axs[0, 0].plot(x[window-1:], np.convolve(y, np.ones(window)/window, mode='valid'), color=color, label=f'{v} ({window}-step mean)')
                axs[1, 0].plot(x, [r['keep'] * 100 for r in rows], color=color, alpha=.7, label=v)
                axs[1, 1].plot(x, [r['router_grad_norm'] for r in rows], color=color, label=f'{v}: routing')
                axs[1, 1].plot(x, [r['head_grad_norm'] for r in rows], '--', color=color, alpha=.6, label=f'{v}: head')
            vp = folder / v / 'curve.json'
            if vp.exists():
                val = json.loads(vp.read_text())
                axs[0, 1].plot([r['updates'] for r in val], [r['nmse'] for r in val], '-o', color=color, label=v)
                rejected = [r for r in val if r.get('eligible') is False]
                if rejected:
                    axs[0, 1].scatter([r['updates'] for r in rejected], [r['nmse'] for r in rejected], marker='x', color='red', s=60, zorder=5)
        for ax, title in zip(axs.flat, ['Train actual hard-kernel NMSE (raw + rolling mean)',
                                      'Validation actual NMSE (all96 events; raw, not best)',
                                      'Train retained KV (%)', 'Gradient norms before clipping']):
            ax.set_title(title)
            ax.set_xlabel('Updates')
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=8)
        axs[1, 0].axhline(15, color='black', linestyle=':')
        axs[1, 1].set_yscale('symlog', linthresh=1e-6)
        fig.suptitle(f'Opt12 joint fine-tuning: layer {layer}; backbone frozen; backward uses STE')
        fig.savefig(destination / f'layer{layer:02d}.png', dpi=150)
        plt.close(fig)
        count += 1
    print(f'{count} layer charts: {destination}')


if __name__ == '__main__':
    plot()
