"""One original scene per authorized GPU; same-device Dense/current pairing."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from ..learned_cache.io import atomic
from ..stream_eval_common import REPO

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['horse_field','traffic_crossing'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output).resolve();out=root/a.case;out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(STREAM_EVAL_ROOT=str(root),STREAM_RESERVATION_PID=str(os.getpid()),OMP_NUM_THREADS='4',PYTHONPATH=str(REPO))
    inputs=REPO/'artifacts/comparison_20261113'/a.case/'shared_inputs.pt'
    for method,module in [('dense','local_experiments.stream_eval_official'),('current_round8_full_pset','local_experiments.video_runtime_v1.comparison_original')]:
        dest=out/method;dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        atomic(out/'status.json',dict(status='running',method=method,pid=os.getpid(),gpu=env['CUDA_VISIBLE_DEVICES']))
        cmd=[sys.executable,'-u','-m',module,'--inputs',str(inputs),'--output',str(dest)]
        if method=='dense':cmd+=['--method','dense']
        with (dest/f'worker.{int(time.time())}.log').open('w') as log:
            result=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            atomic(out/'status.json',dict(status='failed',method=method,exit_code=result.returncode));raise SystemExit(result.returncode)
    atomic(out/'status.json',dict(status='complete'))
if __name__=='__main__':main()
