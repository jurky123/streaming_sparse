"""Quality-only serial worker. Own reservation, isolated child processes.

Shared GPU permitted for quality, never speed acceptance. No foreign process
signals, scheduler locks, or persistent changes to NVIDIA configuration.
"""
import gc,json,os,signal,subprocess,sys,time
from pathlib import Path
import torch
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .gap_gpu_processes import inventory
ROOT=Path('artifacts/review_v1_20261102');BASE='artifacts/review_v1_20261102'

def main():
    lock=RunLock(ROOT/'worker');stop=[False];child=None
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    audit=json.loads((ROOT/'preflight_audit.json').read_text());assert audit['ready_for_pilot']
    for path,sha in audit['code'].items():assert digest(path)==sha,path
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu=='7'
    free,total=torch.cuda.mem_get_info();headroom=60*2**30
    reservation=torch.empty(max(0,free-headroom),dtype=torch.uint8,device='cuda')
    own=dict(pid=os.getpid(),gpu=int(gpu),reservation_bytes=reservation.numel(),working_headroom_bytes=headroom,shared_quality_only=True,speed_valid=False)
    atomic(ROOT/'worker/reservation.json',own)
    jobs=[('gpu_preflight','gap_review_v1_preflight',[]),
          ('probe_initial','gap_review_v1_probe',['--case','bird_flight']),
          ('summary_pilot','gap_review_v1_train',[]),
          ('probe_trained','gap_review_v1_probe',['--case','bird_flight','--tag','summary_pilot','--summary-checkpoint',BASE+'/summary_train/pilot/latest.pt']),
          ('calibration_data','gap_review_v1_probe',['--case','city_walk','--tag','summary_pilot','--summary-checkpoint',BASE+'/summary_train/pilot/latest.pt']),
          ('calibration_fit','gap_review_v1_calibrate',[]),
          ('probe_route_trained','gap_review_v1_probe',['--case','bird_flight','--tag','route_pilot','--summary-checkpoint',BASE+'/route_pilot/model.pt'])]
    try:
        for name,module,args in jobs:
            result=ROOT/'worker'/f'{name}.json'
            if result.exists() and json.loads(result.read_text()).get('status')=='complete':continue
            if stop[0]:break
            for path,sha in audit['code'].items():assert digest(path)==sha,path
            disk_guard(ROOT,50,2)
            while torch.cuda.mem_get_info()[0]<32*2**30 and not stop[0]:
                atomic(ROOT/'worker/status.json',own|dict(status='waiting_for_headroom',next=name,updated=time.time()));time.sleep(5)
            if stop[0]:break
            peers=[p for ps in inventory(gpu).values() for p in ps if p['pid']!=os.getpid()]
            command=[sys.executable,'-u','-m','local_experiments.'+module]+args
            env=os.environ.copy();env.update(GAP_TAIL_MODE='1',PYTHONUNBUFFERED='1')
            with (ROOT/'worker'/f'{name}.log').open('a') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic(ROOT/'worker/status.json',own|dict(status='running',job=name,child_pid=child.pid,peers=peers,command=command,updated=time.time()))
                while child.poll() is None and not stop[0]:time.sleep(2)
                if stop[0] and child.poll() is None:child.terminate()
                code=child.wait();child=None
            atomic(result,dict(status='complete' if code==0 else 'failed',returncode=code,peers=peers,speed_valid=False,updated=time.time()))
            if code:raise RuntimeError(f'{name} exited {code}; inspect worker/{name}.log')
            gc.collect();torch.cuda.empty_cache()
        atomic(ROOT/'worker/status.json',own|dict(status='interrupted' if stop[0] else 'pilot_complete_pending_analysis',updated=time.time()))
    except BaseException as exc:
        atomic(ROOT/'worker/status.json',own|dict(status='failed',error=repr(exc),updated=time.time()));raise
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait()
        lock.close()
if __name__=='__main__':main()
