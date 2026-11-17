"""Finish authorized short/15s Dense comparisons; no 30s jobs."""
import json,os,subprocess,sys,time
from .weights import ROOT
from ..learned_cache.io import atomic,RunLock

def run():
    base=ROOT/'artifacts/video_runtime_v1/formal_short_15';lock=RunLock(base/'finish');env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4')
    def status(stage):atomic(base/'finish/status.json',dict(stage=stage,pid=os.getpid(),updated=time.time()))
    def execute(cmd,path):
        with path.open('a') as log:subprocess.run([sys.executable,'-u','-m',*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        root=ROOT/'artifacts/video_runtime_v1/formal_15_30/15s';dest=root/'native_dense';status('waiting_native_dense15')
        while not (dest/'result.json').exists():time.sleep(5)
        if not (dest/'decode.json').exists():
            status('decode_dense15');execute(['local_experiments.video_runtime_v1.formal_quality','--mode','decode','--root',str(dest)],dest/'decode.log')
        status('quality15');execute(['local_experiments.video_runtime_v1.formal_quality','--mode','metrics','--root',str(root)],root/'quality.log')
        status('short_optimized');execute(['local_experiments.video_runtime_v1.formal_worker','--gpu','7'],base/'worker7.log')
        status('waiting_short_reference')
        while not (base/'short/native_dense/decode.json').exists():
            rec=json.loads((base/'worker6/status.json').read_text());assert rec['stage']!='failed',rec;time.sleep(5)
        status('quality_short');execute(['local_experiments.video_runtime_v1.formal_quality','--mode','metrics','--root',str(base/'short')],base/'short/quality.log');status('complete')
    except BaseException as exc:status('failed: '+repr(exc));raise
    finally:lock.close()
if __name__=='__main__':run()
