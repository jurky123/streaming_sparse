"""Resumable training -> early quality -> gated full-speed acceptance controller."""
import json,os,signal,subprocess,sys
from pathlib import Path
from .learned_cache.io import atomic,RunLock
from .full_route_report import main as report
ROOT=Path('artifacts/full_route_grad_20261031')
def main():
    lock=RunLock(ROOT/'pipeline');stop=[False];child=[None]
    def halt(*_):
        stop[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
    def run(module,extra=()):
        atomic(ROOT/'pipeline'/'status.json',dict(status='running',stage=module,pid=os.getpid()))
        child[0]=subprocess.Popen(['bash','local_setup/run.sh','-m',module,*extra])
        while True:
            try:code=child[0].wait(timeout=30);break
            except subprocess.TimeoutExpired:report()
        report();assert code==0,(module,code)
    try:
        assert json.loads(Path('artifacts/route_graph_20261031/acceptance/status.json').read_text())['status']=='complete'
        assert json.loads((ROOT/'smoke_chunk5.json').read_text())['passed']
        run('local_experiments.full_route_train')
        if stop[0] or json.loads((ROOT/'status.json').read_text())['status']!='training_validation_complete':return
        run('local_experiments.full_route_accept')
        if stop[0]:return
        passed=True
        for case in ['basketball_court','butterfly_garden']:
            folder=ROOT/'acceptance'/'early'/case
            q={m:json.loads((folder/(m+'.json')).read_text())['quality'] for m in ['off','initial','trained']}
            passed &= all(q['trained']['lpips']<q[m]['lpips'] and q['trained']['ssim']>=q[m]['ssim'] for m in ['off','initial'])
        atomic(ROOT/'quality_gate.json',dict(passed=bool(passed),rule='both scenes: trained LPIPS lower and SSIM no worse than both initial and off; early RGB[21,101)'))
        if passed:run('local_experiments.full_route_accept',['--phase','full','--modes','initial,trained'])
        atomic(ROOT/'pipeline'/'status.json',dict(status='completed',quality_gate=bool(passed),full_speed_run=bool(passed)));report()
    except Exception as exc:atomic(ROOT/'pipeline'/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if stop[0]:atomic(ROOT/'pipeline'/'status.json',dict(status='interrupted'))
        lock.close()
if __name__=='__main__':main()
