"""Same-runtime 2x2 weight/module ablation using the original comparison videos."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from ..learned_cache.io import atomic,digest
from ..stream_eval_common import REPO

ROOT=REPO/'artifacts/weight_module_ablation_20261117'
PRIOR=REPO/'artifacts/comparison_current_baseline_20261117'
VARIANTS=[('old_summary_risk','old',False),('new_summary_risk','new',False),('old_summary_full_pset','old',True)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['horse_field','traffic_crossing'],required=True);a=p.parse_args()
    out=ROOT/a.case;out.mkdir(parents=True,exist_ok=True)
    for method in ['dense','current_round8_full_pset']:
        link=out/method
        if not link.exists():link.symlink_to(PRIOR/a.case/method,target_is_directory=True)
    env=os.environ.copy();env.update(STREAM_EVAL_ROOT=str(ROOT),STREAM_RESERVATION_PID=str(os.getpid()),OMP_NUM_THREADS='4',PYTHONPATH=str(REPO))
    inputs=REPO/'artifacts/comparison_20261113'/a.case/'shared_inputs.pt'
    for method,summary,full in VARIANTS:
        dest=out/method;dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        atomic(out/'status.json',dict(status='running',method=method,pid=os.getpid(),gpu=env['CUDA_VISIBLE_DEVICES']))
        cmd=[sys.executable,'-u','-m','local_experiments.video_runtime_v1.comparison_original','--inputs',str(inputs),'--output',str(dest),'--summary',summary]
        if not full:cmd+=['--without-full-pset']
        with (dest/f'worker.{int(time.time())}.log').open('w') as log:
            result=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            atomic(out/'status.json',dict(status='failed',method=method,exit_code=result.returncode));raise SystemExit(result.returncode)
    atomic(out/'status.json',dict(status='complete'))
if __name__=='__main__':main()
