"""Reduce the paired timings and device profiles, keeping old and new measurements separate."""
import json
from pathlib import Path
from statistics import median

directory=Path(__file__).parent
def read(name):
    return json.loads((directory/(name+'.json')).read_text())

pairs={}
for name in ('floor_center_video','fused_center_video','late_mask_video','late_mask_multi'):
    rows=read(name)
    gains=[1000*(r['baseline_s']-r['candidate_s']) for r in rows]
    pairs[name]=dict(n=len(gains), wins=sum(x>0 for x in gains), median_saved_ms=median(gains),
                     min_saved_ms=min(gains), max_saved_ms=max(gains),
                     speedup=median(r['baseline_s']/r['candidate_s'] for r in rows))
profiles={}
for name in ('control','fused_center','floor_center'):
    chunks=read(name+'_timeline')
    profile={prefix:sum(ms for chunk in chunks for kernel,ms in chunk['by_name_ms'].items()
                        if kernel.startswith(prefix)) for prefix in ('exact(','prepare(','retry')}
    total=sum(profile.values())
    profiles[name]=dict(main_ms=profile['exact('],prepare_ms=profile['prepare('],retry_ms=profile['retry'],
                        total_ms=total,main_tflops=147510/profile['exact('],total_tflops=147510/total)
output=dict(pairs=pairs,device=profiles,
            tiers={k:read('tier_'+k)['median_s'] for k in ('base','intersection','step')},
            guard={k:{m:r[m] for m in ('fallback_tiles','calls_with_fallback')}
                   for k,r in read('trajectory_guard').items()})
(directory/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
