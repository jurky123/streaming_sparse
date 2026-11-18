"""Aggregate paired wall clocks, CUDA stages, and three same-card release runs."""
from collections import Counter
import json
from pathlib import Path
from statistics import median
import torch

folder = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text())

result = {}
for name in ('shared_video','shared_multi','release_video','release_multi','final_video'):
    rows = read(folder / (name+'.json'))
    multi = name.endswith('multi')
    assert len(rows) == (10 if multi else 8)
    assert len({(r['prompt'],r['seed']) for r in rows}) == (10 if multi else 1)
    assert all(r['bitwise'] for r in rows)
    deltas = [1000*(r['baseline_s']-r['candidate_s']) for r in rows]
    result[name] = dict(pairs=len(rows),faster=sum(v>0 for v in deltas),median_ms=median(deltas),
                        range_ms=[min(deltas),max(deltas)],speedup=median(r['baseline_s']/r['candidate_s'] for r in rows))
for name in ('baseline','shared','release'):
    kernels = Counter()
    for chunk in read(folder / f'timeline_{name}.json'):
        kernels.update(chunk['by_name_ms'])
    def total(*prefixes):
        return sum(ms for k,ms in kernels.items() if k.startswith(prefixes))
    result[name+'_ms'] = dict(rope=total('void rms_row_kernel<true'),prepare=total('prepare(', 'prepare_current(', 'void prepare_packed_k'),
                              main=total('exact('),retry=total('retry'))
dense = torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True).float()
for tier in ('base','intersection','step'):
    run = Path('/tmp/jzk_rope_tier_'+tier)
    prev = 'deployed_base' if tier=='base' else tier
    z = torch.load(run/'latents.pt',weights_only=True)
    old = torch.load(Path('/tmp/jzk_history_summary_'+prev)/'latents.pt',weights_only=True)
    assert torch.equal(z.view(torch.uint8),old.view(torch.uint8))
    r = read(run/'result.json')
    assert r['runtime_stats']['packed_attention_calls'] == (812 if tier=='base' else 0)
    result[tier] = dict(samples=r['seconds'],seconds=r['median_seconds'],peak_bytes=r['peak_allocated_bytes'],
                        bitwise=True,dense_nmse=float((z.float()-dense).square().sum()/dense.square().sum()))
(folder/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
