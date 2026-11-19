"""Persistent supervisor: 4 training GPUs, 2 validation GPUs; leave 6/7 free."""
import os,subprocess,sys,time
from .common import ROOT,OUT,atomic
from ..learned_cache.io import RunLock


def run():
    lock=RunLock(OUT/'full_pset_pipeline');jobs={};evaluation={};done=set()
    try:
        for seed,gpus in ((17,'0,1'),(29,'2,3')):
            root=OUT/'training'/f'Full_Pset_s{seed}';root.mkdir(parents=True,exist_ok=True)
            if (root/'complete.json').exists():done.add(seed);continue
            if (root/'training_complete.json').exists():continue
            cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2','-m','local_experiments.full_coverage.train_full_pset','--seed',str(seed)]
            with (root/'run.log').open('a') as log:jobs[seed]=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while len(done)<2:
            for seed,gpu in ((17,'4'),(29,'5')):
                if seed in done:continue
                root=OUT/'training'/f'Full_Pset_s{seed}'
                if seed in jobs and jobs[seed].poll() not in (None,0):raise RuntimeError(f'Full_Pset seed{seed} training failed; see {root}/run.log')
                if (root/'training_complete.json').exists() and (seed not in jobs or jobs[seed].poll()==0) and seed not in evaluation:
                    cmd=[sys.executable,'-u','-m','local_experiments.full_coverage.train_full_pset','--seed',str(seed),'--evaluate']
                    with (root/'evaluation.log').open('a') as log:evaluation[seed]=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                if seed in evaluation and evaluation[seed].poll() is not None:
                    if evaluation[seed].returncode:raise RuntimeError(f'Full_Pset seed{seed} validation failed')
                    assert (root/'complete.json').exists();done.add(seed)
            atomic(OUT/'full_pset_pipeline/status.json',dict(stage='complete' if len(done)==2 else 'Full_Pset_training_or_validation',training_gpus=[0,1,2,3],validation_gpus=[4,5],free_gpus=[6,7],training_pids={s:p.pid for s,p in jobs.items() if p.poll() is None},evaluation_pids={s:p.pid for s,p in evaluation.items() if p.poll() is None},completed=sorted(done),pid=os.getpid(),updated=time.time()))
            if len(done)<2:time.sleep(10)
    except BaseException as exc:
        atomic(OUT/'full_pset_pipeline/status.json',dict(stage='failed',error=repr(exc),pid=os.getpid(),updated=time.time()));raise
    finally:lock.close()
if __name__=='__main__':run()
