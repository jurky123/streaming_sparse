"""Summarize paired videos, kernel timelines, and the three normal tier benchmarks."""
import json
from pathlib import Path
from statistics import median

import torch

root=Path(__file__).parent
def read(path):
    return json.loads(path.read_text())

summary={}
for candidate in ('cross_phase','last_page'):
    summary[candidate]={}
    for which,size in [('video',8),('multi',10)]:
        records=read(root/f'{candidate}_{which}.json')
        assert len(records)==size and all(row['bitwise'] for row in records)
        if which=='multi':
            assert len({(r['prompt'],r['seed']) for r in records})==10
        savings=[1000*(r['baseline_s']-r['candidate_s']) for r in records]
        summary[candidate][which]=dict(faster=sum(x>0 for x in savings),pairs=size,
            paired_ms=median(savings),range_ms=[min(savings),max(savings)],
            speedup=median(r['baseline_s']/r['candidate_s'] for r in records))
for name in ('release','last_page'):
    rows=read(root/f'timeline_{name}.json')
    assert len(rows)==9
    times={op:sum(ms for r in rows for k,ms in r['by_name_ms'].items() if k.startswith(op))
           for op in ('exact(','prepare(','retry')}
    main=times['exact(']
    summary.setdefault(name,{})['device']=dict(main_ms=main,main_tflops=147510/main,
        gap_ms=main-712.914,prepare_ms=times['prepare('],retry_ms=times['retry'])
summary['tiers']={}
for tier in ('base','intersection','step'):
    path=Path('/tmp/jzk_lastpage_tier_'+tier)
    result=read(path/'result.json')
    z=torch.load(path/'latents.pt',weights_only=True)
    old=torch.load(Path('/tmp/jzk_vector_deployed_'+tier)/'latents.pt',weights_only=True)
    assert torch.equal(z.view(torch.uint8),old.view(torch.uint8))
    calls=result['runtime_stats']['packed_attention_calls']
    assert calls==(812 if tier=='base' else 0)
    summary['tiers'][tier]=dict(seconds=result['seconds'],median_seconds=result['median_seconds'],
                                cuda_calls=calls,bitwise=True)
official=Path('/tmp/jzk_lastpage_official_base')
r=read(official/'result.json')
z=torch.load(official/'latents.pt',weights_only=True)
old=torch.load('/tmp/jzk_vector_deployed_base/latents.pt',weights_only=True)
dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True).float()
assert torch.equal(z.view(torch.uint8),old.view(torch.uint8))
assert r['runtime_stats']['packed_attention_calls']==812
summary['official_base']=dict(seconds=r['seconds'],median_seconds=r['median_seconds'],bitwise=True,
    dense_nmse=float((z.float()-dense).square().sum()/dense.square().sum()))
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
