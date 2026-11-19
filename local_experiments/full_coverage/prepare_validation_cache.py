"""Fill reusable frozen OldRisk evaluation once on the six idle GPUs."""
import argparse,json,os,subprocess,sys,time
from .common import ROOT,OUT,atomic
from .evaluate_parallel import CACHE,merge_baseline
from ..learned_cache.io import RunLock

def run(gpus):
    CACHE.mkdir(parents=True,exist_ok=True);lock=RunLock(CACHE/'coordinator');children=[]
    try:
        for shard,gpu in enumerate(gpus):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,'-u','-m','local_experiments.full_coverage.evaluate_parallel','--mode','baseline','--shard',str(shard),'--shards',str(len(gpus)),'--output',str(CACHE/f'part{shard}.json')]
            with (CACHE/f'part{shard}.log').open('a') as log:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p)
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Frozen baseline worker failed')
            atomic(CACHE/'status.json',dict(status='building',completed=sum((CACHE/f'part{s}.json').exists() for s in range(len(gpus))),shards=len(gpus),pids=[p.pid for p in children],updated=time.time()));time.sleep(10)
        if any(p.returncode for p in children):raise RuntimeError('Frozen baseline worker failed')
        merge_baseline(len(gpus));atomic(CACHE/'status.json',dict(status='complete',events=8352,updated=time.time()))
    finally:lock.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpus',type=int,nargs='+',default=[2,3,4,5,6,7]);a=p.parse_args();run(a.gpus)
