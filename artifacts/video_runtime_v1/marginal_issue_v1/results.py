"""Collect matched wall times and validate all default-tier latents against the last release."""
import json
from pathlib import Path
from statistics import median
import torch

root = Path(__file__).resolve().parent
def load(path):
    return json.loads(path.read_text())

summary = {}
for group, expected in [('video', 8), ('repeat_video', 8), ('multi', 10)]:
    rows = load(root / f'coeffreuse_{group}.json')
    assert len(rows) == expected and all(r['bitwise'] for r in rows)
    if group == 'multi':
        assert len(set(r['prompt'] for r in rows)) == 5
        assert len(set(r['seed'] for r in rows)) == 2
        assert len(set((r['prompt'], r['seed']) for r in rows)) == 10
    changes = [1000 * (r['baseline_s'] - r['candidate_s']) for r in rows]
    summary[group] = dict(faster=sum(x > 0 for x in changes), pairs=len(rows), median_ms=median(changes),
                          range_ms=[min(changes), max(changes)],
                          speedup=median(r['baseline_s']/r['candidate_s'] for r in rows))
for variant in ['base', 'coeffreuse']:
    rows = load(root / f'timeline_{variant}.json')
    summary[variant + '_device_ms'] = sum(t for r in rows for k,t in r['by_name_ms'].items() if k.startswith('exact('))
dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True).float()
for tier in ['base','intersection','step']:
    current = Path('/tmp/jzk_marginal_control_' + tier)
    if tier == 'base':
        current = Path('/tmp/jzk_marginal_release_base')
    previous = Path('/tmp/jzk_cursor_control_' + tier)
    a, b = [torch.load(p/'latents.pt', weights_only=True) for p in (current, previous)]
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    run=load(current/'result.json')
    assert run['runtime_stats']['packed_attention_calls'] == (812 if tier == 'base' else 0)
    samples = run['seconds']
    summary[tier] = dict(median_s=median(samples), seconds=samples, bitwise_to_release=True, dense_nmse=float((a.float()-dense).square().sum()/dense.square().sum()))
(root/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
print(summary)
