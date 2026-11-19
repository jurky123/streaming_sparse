"""Continue both seeds; reserve spare GPUs for immutable-checkpoint evaluation."""
import argparse,os,subprocess,sys,time
from .common import ROOT,OUT,atomic
from .audit import export
from ..learned_cache.io import RunLock

def run(validated_execution_upgrade=False):
    lock=RunLock(OUT/'pipeline');children=[]
    try:
        for seed,gpu,evaluation in ((17,0,[2,3,4]),(29,1,[5,6,7])):
            root=OUT/'training'/f'C_s{seed}'
            if (root/'complete.json').exists():continue
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,'-u','-m','local_experiments.full_coverage.train_optimized','--seed',str(seed),'--eval-gpus',*map(str,evaluation)]
            if validated_execution_upgrade:cmd.append('--validated-execution-upgrade')
            with (root/'run_optimized.log').open('a') as log:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p)
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Optimized training failed; latest checkpoints preserved')
            atomic(OUT/'pipeline/status.json',dict(stage='C_training',implementation='prefetch_shared_risk_async_eval_v1',training_gpus=[0,1],validation_gpus=[2,3,4,5,6,7],pids=[p.pid for p in children],pid=os.getpid(),updated=time.time()));time.sleep(20)
        if any(p.returncode for p in children):raise RuntimeError('Optimized training failed')
        for seed in (17,29):export(seed)
        atomic(OUT/'pipeline/status.json',dict(stage='C_complete',positions=348,seeds=[17,29],full_pset_ready=False,updated=time.time()))
    except BaseException as exc:
        atomic(OUT/'pipeline/status.json',dict(stage='failed',error=repr(exc),pid=os.getpid(),updated=time.time()));raise
    finally:lock.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validated-execution-upgrade',action='store_true');a=p.parse_args();run(a.validated_execution_upgrade)
