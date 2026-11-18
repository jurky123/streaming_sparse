"""Summarize paired timings and verify formal tier outputs against the previous release."""
import json
from pathlib import Path
from statistics import median
import torch
root=Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text())
summary={}
for name,n in [('video',8),('multi',10)]:
    records=read(root/f'maskreg_{name}.json')
    assert len(records)==n and all(r['bitwise'] for r in records)
    if name=='multi': assert len({(r['prompt'],r['seed']) for r in records})==10
    change=[1000*(r['baseline_s']-r['candidate_s']) for r in records]
    summary[name]=dict(n=n,faster=sum(x>0 for x in change),median_ms=median(change),range_ms=[min(change),max(change)],
                       speedup=median(r['baseline_s']/r['candidate_s'] for r in records))
for name in ('base','maskreg'):
    times=read(root/f'timeline_{name}.json')
    summary[name+'_kernel_ms']=sum(v for c in times for k,v in c['by_name_ms'].items() if k.startswith('exact('))
for name in ('maskreg','maskreg_repeat'):
    saving=0
    for p,n in [(216,116),(288,116),(360,116),(432,464)]:
        r=read(root/f'{name}_{p}.json'); assert r['bitwise']
        saving+=n*median(a-b for a,b in zip(*r['samples']))
    summary[name+'_weighted_ms_saved']=saving
for tier in ('base','intersection','step'):
    path=Path('/tmp/jzk_cursor_control_'+tier)
    run=read(path/'result.json')
    z=torch.load(path/'latents.pt',weights_only=True)
    prior=torch.load(Path('/tmp/jzk_rope_tier_'+tier)/'latents.pt',weights_only=True)
    assert torch.equal(z.view(torch.uint8),prior.view(torch.uint8))
    summary['control_'+tier]=dict(samples=run['seconds'],seconds=run['median_seconds'],bitwise=True)
(root/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary)
