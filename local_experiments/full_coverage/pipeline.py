"""Durable orchestration: capture -> coverage audit -> two C seeds -> export.

Uses OS processes, one resident capture model per GPU. A failed stage stops the
pipeline with its log location. Re-running resumes complete captures/checkpoints.
"""
import argparse,json,os,subprocess,sys,time
from .common import *
from ..learned_cache.io import RunLock

def active(pid):
    try:os.kill(pid,0);return True
    except (ProcessLookupError,TypeError):return False

def start(module,args,gpu,folder):
    folder.mkdir(parents=True,exist_ok=True);env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2')
    cmd=[sys.executable,'-u','-m',module]+list(args)
    with (folder/'run.log').open('a') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return p

def run(gpus):
    p=protocol();lock=RunLock(OUT/'pipeline');children=[]
    try:
        for worker,gpu in enumerate(gpus):
            folder=OUT/f'workers/capture{worker}';status=folder/'status.json'
            old=json.loads(status.read_text()) if status.exists() else {}
            # Adopt only this pipeline's exact capture command, never a foreign PID.
            pid=old.get('pid');cmdline=Path(f'/proc/{pid}/cmdline')
            owned=active(pid) and cmdline.exists() and b'local_experiments.full_coverage.capture' in cmdline.read_bytes()
            if not owned:
                children.append(start('local_experiments.full_coverage.capture',['--worker',str(worker),'--workers',str(len(gpus))],gpu,folder))
        while True:
            done=sum((OUT/'trajectories'/c['id']/'complete.json').exists() for c in p['cases'])
            atomic(OUT/'pipeline/status.json',dict(stage='capture',completed=done,total=len(p['cases']),pid=os.getpid(),updated=time.time()))
            if done==len(p['cases']):break
            for worker in range(len(gpus)):
                status=OUT/f'workers/capture{worker}/status.json'
                if status.exists() and json.loads(status.read_text()).get('status')=='failed':
                    raise RuntimeError(f'Capture {worker} failed: {status}; see sibling run.log')
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Capture process failed; inspect workers logs')
            time.sleep(20)
        for c in children:c.wait()
        from .audit import audit_data
        audit_data();children=[]
        for seed,gpu in zip((17,29),(gpus[0],gpus[1] if len(gpus)>1 else gpus[0])):
            if len(gpus)==1 and children:children[-1].wait()
            root=OUT/'training'/f'C_s{seed}'
            if not (root/'complete.json').exists():children.append(start('local_experiments.full_coverage.train',['--seed',str(seed)],gpu,root))
        while any(c.poll() is None for c in children):
            atomic(OUT/'pipeline/status.json',dict(stage='C_training',pid=os.getpid(),updated=time.time()))
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('C training failed; inspect training/C_s*/run.log')
            time.sleep(20)
        if any(c.returncode for c in children):raise RuntimeError('C training failed')
        from .audit import export
        for seed in (17,29):export(seed)
        atomic(OUT/'pipeline/status.json',dict(stage='C_complete',positions=348,seeds=[17,29],full_pset_ready=False,updated=time.time()))
    except BaseException as exc:
        atomic(OUT/'pipeline/status.json',dict(stage='failed',error=repr(exc),pid=os.getpid(),updated=time.time()));raise
    finally:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpus',type=int,nargs='+',required=True);a=p.parse_args()
    if len(a.gpus)!=len(set(a.gpus)):raise ValueError('Duplicate GPU')
    run(a.gpus)
