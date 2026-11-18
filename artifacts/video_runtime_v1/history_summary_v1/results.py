"""Summarize the resident-summary cache against the unchanged attention kernel."""
import json
from pathlib import Path
from statistics import median
import torch

root=Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text())

result={}
for name in ('cached','release'):
    for group,n in [('video',8),('multi',10)]:
        records=read(root/f'{name}_{group}.json')
        assert len(records)==n and all(r['bitwise'] and r['cache_hits']==609 for r in records)
        if group=='multi':assert len({(r['prompt'],r['seed']) for r in records})==10
        delta=[1000*(r['baseline_s']-r['candidate_s']) for r in records]
        result[f'{name}_{group}']=dict(pairs=n,faster=sum(t>0 for t in delta),median_ms=median(delta),
            min_ms=min(delta),max_ms=max(delta),speedup=median(r['baseline_s']/r['candidate_s'] for r in records))
for name in ('release','cached','deployed'):
    rows=read(root/f'timeline_{name}.json')
    assert len(rows)==9
    times={prefix:sum(v for c in rows for k,v in c['by_name_ms'].items() if k.startswith(prefix))
           for prefix in ('exact(','prepare(','prepare_current(','retry')}
    result[f'{name}_device']=dict(main_ms=times['exact('],prepare_ms=times['prepare(']+times['prepare_current('],
        retry_ms=times['retry'],total_ms=sum(times.values()),main_tflops=147142.60999/times['exact('])
result['tiers']={}
dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True).float()
for mode in ('base','intersection','step','release'):
    tier=mode if mode!='release' else 'base'
    path=Path('/tmp/jzk_history_summary_'+('deployed_base' if mode=='base' else mode))
    r=read(path/'result.json')
    stats=r['runtime_stats']
    z=torch.load(path/'latents.pt',weights_only=True)
    old=torch.load(Path('/tmp/jzk_anchorhalf_'+tier)/'latents.pt',weights_only=True)
    assert torch.equal(z.view(torch.uint8),old.view(torch.uint8))
    assert stats['packed_attention_calls']==(812 if tier=='base' else 0)
    if tier=='base': assert stats['packed_summary_hits']==609 and stats['packed_summary_bytes']==199646208
    result['tiers'][mode]=dict(seconds=r['seconds'],median_seconds=r['median_seconds'],
        peak_allocated_bytes=r['peak_allocated_bytes'],bitwise=True,dense_nmse=float((z.float()-dense).square().sum()/dense.square().sum()))
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
