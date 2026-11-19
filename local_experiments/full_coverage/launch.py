"""Launch capture processes on explicitly selected GPUs; record owned PIDs only."""
import argparse,json,os,subprocess,sys
from .common import OUT,protocol
from ..learned_cache.io import atomic

def launch(gpus):
    protocol();rows=[]
    for worker,gpu in enumerate(gpus):
        folder=OUT/'workers'/f'capture{worker}';folder.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2')
        command=[sys.executable,'-u','-m','local_experiments.full_coverage.capture','--worker',str(worker),'--workers',str(len(gpus))]
        with (folder/'run.log').open('a') as log:
            p=subprocess.Popen(command,cwd=OUT.parents[1],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rows.append(dict(worker=worker,gpu=gpu,pid=p.pid,command=command))
    atomic(OUT/'workers/launch.json',rows);print(json.dumps(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpus',type=int,nargs='+',required=True);a=p.parse_args()
    if len(set(a.gpus))!=len(a.gpus):raise ValueError('Duplicate GPU')
    launch(a.gpus)
