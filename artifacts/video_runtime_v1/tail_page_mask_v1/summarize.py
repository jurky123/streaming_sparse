"""Summarize like-for-like paired wall times and the three device profiles."""
import json
from pathlib import Path
from statistics import median

root=Path(__file__).parent
pairs={}
for name in ('lane_mask_video','lane_mask_multi','lane_fma_video'):
    rows=json.loads((root/(name+'.json')).read_text())
    delta=[1000*(r['baseline_s']-r['candidate_s']) for r in rows]
    pairs[name]=dict(pairs=len(rows),faster=sum(d>0 for d in delta),median_saved_ms=median(delta),
                     min_saved_ms=min(delta),max_saved_ms=max(delta),all_bitwise=all(r['bitwise'] for r in rows))
device={}
for name in ('control','lane_fma','release'):
    rows=json.loads((root/(name+'_timeline.json')).read_text())
    times={label:sum(ms for chunk in rows for k,ms in chunk['by_name_ms'].items() if k.startswith(prefix))
           for label,prefix in [('main','exact('),('prepare','prepare('),('guard','retry')]}
    total=sum(times.values())
    device[name]=dict(ms=times,total_ms=total,main_tflops=147510/times['main'],total_tflops=147510/total,
                      main_gap_ms=times['main']-712.914,total_gap_ms=total-712.914)
(root/'comparison.json').write_text(json.dumps(dict(pairs=pairs,device=device),indent=2)+'\n')
