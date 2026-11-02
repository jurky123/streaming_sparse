"""Integration checks: real data, owned child signals, exact resume, corruption."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import torch
from .learned_cache.io import restore, RunLock, atomic

ROOT=Path('artifacts/learned_cache_pilot_20261029/recovery')
ROOT.mkdir(parents=True,exist_ok=True)

def launch(name,extra=()):
    return subprocess.Popen([sys.executable,'-m','local_experiments.learned_cache.run','--root',str(ROOT/name),
        '--stage','prototype','--steps','6',*extra],stdout=(ROOT/(name+'.log')).open('w'),stderr=subprocess.STDOUT)

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    assert launch('continuous').wait()==0
    assert launch('split',['--max-updates','3']).wait()==0
    assert json.loads((ROOT/'split/status.json').read_text())['status']=='interrupted'
    assert launch('split').wait()==0
    a,_=restore(ROOT/'continuous');b,_=restore(ROOT/'split')
    result={k:equal(a[k],b[k]) for k in ('model','corrector','optimizer','python_rng','torch_rng','cuda_rng','step','history')}
    assert all(result.values()),result
    corrupt=ROOT/'corrupt';corrupt.mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'split/latest.pt',corrupt/'previous.pt')
    (corrupt/'latest.pt').write_bytes(b'broken');(corrupt/'next.pt.tmp').write_bytes(b'partial')
    recovered,name=restore(corrupt);assert name=='previous.pt' and equal(recovered['model'],b['model'])
    result['corrupt_latest_and_partial_temp']=True
    lock=RunLock(ROOT/'lock')
    try:
        try:RunLock(ROOT/'lock')
        except RuntimeError:result['exclusive_lock']=True
        else:raise AssertionError('Lock not exclusive')
    finally:lock.close()
    child=launch('signal')
    deadline=time.monotonic()+120
    while not (ROOT/'signal/latest.pt').exists():
        if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Signal worker startup failed')
        time.sleep(.1)
    os.kill(child.pid,signal.SIGTERM)
    assert child.wait(timeout=60)==0
    status=json.loads((ROOT/'signal/status.json').read_text());assert status['status']=='interrupted',status
    assert restore(ROOT/'signal')[0] is not None
    result['sigterm_checkpoint']=True
    assert launch('signal').wait()==0
    c,_=restore(ROOT/'signal');assert equal(c['model'],a['model']) and equal(c['optimizer'],a['optimizer'])
    result['sigterm_resume_exact']=True
    atomic(ROOT/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
