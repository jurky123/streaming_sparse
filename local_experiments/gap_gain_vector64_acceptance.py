"""Gate the expensive acceptance sequence on clean complete-route improvement."""
import gc,json,runpy,statistics,torch
from pathlib import Path
if __name__=='__main__':
    runpy.run_module('local_experiments.gap_gain_vector64_check',run_name='__main__')
    root=Path('artifacts/gap_gain_vector64_20261102')
    d=json.loads((root/'checks.json').read_text())
    assert d['status']=='complete' and d['peer_monitor_validated']
    means={name:statistics.mean(t['wall_ms'] for r in d['rows'] for t in r['trials'] if t['variant']==name) for name in ['before','after']}
    assert means['after']<means['before'], means
    for mod in ['gap_gain_vector64_quality','gap_gain_vector64_perf']:
        gc.collect();torch.cuda.empty_cache()
        runpy.run_module('local_experiments.'+mod,run_name='__main__')
