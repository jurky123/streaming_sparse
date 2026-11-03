"""Quality-only serial worker. Own reservation, isolated child processes.

Shared GPU permitted for quality, never speed acceptance. No foreign process
signals, scheduler locks, or persistent changes to NVIDIA configuration.
"""
import argparse,gc,json,os,signal,subprocess,sys,time
from pathlib import Path
import torch
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .gap_gpu_processes import inventory
ROOT=Path('artifacts/review_v1_20261102');BASE='artifacts/review_v1_20261102'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args()
    work=ROOT/'round2_worker';lock=RunLock(work);stop=[False];child=None
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu=='7'
    free,total=torch.cuda.mem_get_info();headroom=60*2**30
    reserve_bytes=max(0,free-headroom);unit=2**30
    reservation=[torch.empty(min(unit,reserve_bytes-i),dtype=torch.uint8,device='cuda') for i in range(0,reserve_bytes,unit)]
    own=dict(pid=os.getpid(),gpu=int(gpu),reservation_bytes=sum(t.numel() for t in reservation),working_headroom_bytes=headroom,shared_quality_only=True,speed_valid=False)
    atomic(work/'reservation.json',own)
    try:
        while not stop[0]:
            jobs=json.loads(Path(args.plan).read_text())
            pending=[j for j in jobs if not (work/(j['id']+'.json')).exists() or json.loads((work/(j['id']+'.json')).read_text()).get('status')!='complete']
            if not pending:break
            job=pending[0];name=job['id'];module=job['module'];jobargs=job.get('args',[])
            result=work/f'{name}.json'
            if result.exists() and json.loads(result.read_text()).get('status')=='complete':continue
            if stop[0]:break
            disk_guard(ROOT,50,4)
            while not job.get('cpu_only',False) and torch.cuda.mem_get_info()[0]<32*2**30 and not stop[0]:
                if reservation:
                    del reservation[-1];torch.cuda.empty_cache();own['reservation_bytes']=sum(t.numel() for t in reservation);continue
                atomic(work/'status.json',own|dict(status='waiting_for_headroom',next=name,updated=time.time()));time.sleep(5)
            if stop[0]:break
            peers=[p for ps in inventory(gpu).values() for p in ps if p['pid']!=os.getpid()]
            command=[sys.executable,'-u','-m','local_experiments.'+module]+jobargs
            env=os.environ.copy();env.update(GAP_TAIL_MODE='1',PYTHONUNBUFFERED='1')
            if job.get('cpu_only',False):env['CUDA_VISIBLE_DEVICES']=''
            with (work/f'{name}.log').open('a') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic(work/'status.json',own|dict(status='running',job=name,child_pid=child.pid,peers=peers,command=command,source_sha256=digest('local_experiments/'+module+'.py'),updated=time.time()))
                while child.poll() is None and not stop[0]:time.sleep(2)
                if stop[0] and child.poll() is None:child.terminate()
                code=child.wait();child=None
                if stop[0] and code==0:code=130
            atomic(result,dict(status='complete' if code==0 else 'failed',returncode=code,peers=peers,speed_valid=False,updated=time.time()))
            if code:raise RuntimeError(f'{name} exited {code}; inspect worker/{name}.log')
            gc.collect();torch.cuda.empty_cache()
        atomic(work/'status.json',own|dict(status='interrupted' if stop[0] else 'round_complete_pending_analysis',updated=time.time()))
    except BaseException as exc:
        atomic(work/'status.json',own|dict(status='failed',error=repr(exc),updated=time.time()));raise
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait()
        lock.close()
if __name__=='__main__':main()
