"""Single-video short/15s worker; 30s runs were cancelled by the user."""
import argparse,os,subprocess,sys
from .weights import ROOT
from ..learned_cache.io import RunLock,atomic

def run(gpu):
    base=ROOT/'artifacts/video_runtime_v1/formal_short_15';root=base/'short';lock=RunLock(base/f'worker{gpu}')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4')
    jobs=[('original','route'),('native_dense','dense_native')] if gpu==6 else [('optimized','action')]
    try:
        for name,mode in jobs:
            dest=root/name;dest.mkdir(parents=True,exist_ok=True)
            if not (dest/'result.json').exists():
                atomic(base/f'worker{gpu}/status.json',dict(stage='generate',method=name,pid=os.getpid()))
                cmd=[sys.executable,'-u','-m','local_experiments.video_runtime_v1.benchmark_hotspots','--mode',mode,'--chunks','9','--history-pages','360','--skip-profile','--output',str(dest),'--reader-checkpoint',str(ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')]
                with (dest/'generate.log').open('a') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not (dest/'decode.json').exists():
                atomic(base/f'worker{gpu}/status.json',dict(stage='decode',method=name,pid=os.getpid()))
                with (dest/'decode.log').open('a') as log:subprocess.run([sys.executable,'-u','-m','local_experiments.video_runtime_v1.formal_quality','--root',str(dest),'--mode','decode'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic(base/f'worker{gpu}/status.json',dict(stage='complete'))
    except BaseException as exc:atomic(base/f'worker{gpu}/status.json',dict(stage='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[6,7],required=True);run(p.parse_args().gpu)
