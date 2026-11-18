"""Separate video wall time, full-path device time and same-shape Dense reference."""
import json
from pathlib import Path
from statistics import median

root=Path(__file__).parent
def read(name):
    return json.loads((root/(name+'.json')).read_text())

device={}
for name in ('control','direct_mass','release','vector_release'):
    rows=read(name+'_timeline')
    times={label:sum(v for c in rows for k,v in c['by_name_ms'].items() if k.startswith(prefix))
           for label,prefix in [('main','exact('),('prepare','prepare('),('guard','retry')]}
    device[name]=dict(ms=times,total_ms=sum(times.values()),main_tflops=147510/times['main'],
                      full_path_tflops=147510/sum(times.values()),
                      main_gap_ms=times['main']-712.914,full_gap_ms=sum(times.values())-712.914)
r=read('direct_mass_video')
saved=[1000*(x['baseline_s']-x['candidate_s']) for x in r]
pairs=dict(pairs=len(r),faster=sum(v>0 for v in saved),median_ms=median(saved),min_ms=min(saved),max_ms=max(saved))
references={}
for name in ('dense','dense_r2','dense_other'):
    total=0
    rows={}
    for p,count in [(216,116),(288,116),(360,116),(432,464)]:
        row=read(f'{name}_{p}')
        total+=count*row['reference_ms']
        rows[p]=dict(fraction=row['throughput_fraction'],delta_us=1000*(row['main_ms']-row['reference_ms']))
    references[name]=dict(total_reference_ms=total,windows=rows)
additional={}
for name in ('pair_store','vector_store','joint_release'):
    trials=read(name+'_video')
    gains=[1000*(r['baseline_s']-r['candidate_s']) for r in trials]
    additional[name]=dict(pairs=len(gains),faster=sum(v>0 for v in gains),median_ms=median(gains),
                          min_ms=min(gains),max_ms=max(gains),bitwise=all(r['bitwise'] for r in trials))
(root/'comparison.json').write_text(json.dumps(dict(device=device,paired_video=pairs,additional_pairs=additional,fresh_references=references),indent=2)+'\n')
