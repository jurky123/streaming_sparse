"""One ablation cell plus its same-device Dense reference."""
import argparse,os,subprocess,sys,time,json
from pathlib import Path
import torch
from ..stream_eval_common import REPO
from ..learned_cache.io import atomic
ROOT=REPO/'artifacts/weight_module_ablation_20261117'

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--method',choices=['new_summary_risk','old_summary_full_pset','current_round8_full_pset'],required=True);a=p.parse_args()
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];base=ROOT/a.case
    inputs=REPO/'artifacts/comparison_20261113'/a.case/'shared_inputs.pt'
    dense=ROOT/'dense_pairs'/f'gpu{gpu}'/a.case/'dense'
    env=os.environ.copy();env.update(STREAM_EVAL_ROOT=str(ROOT),STREAM_RESERVATION_PID=str(os.getpid()),OMP_NUM_THREADS='4',PYTHONPATH=str(REPO))
    def run(module,dest,args):
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():return
        with (dest/f'worker.{int(time.time())}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u','-m',module,'--inputs',str(inputs),'--output',str(dest),*args],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
    run('local_experiments.stream_eval_official',dense,['--method','dense'])
    torch.set_num_threads(2)
    ref=torch.load(base/'dense/latents.pt',map_location='cpu',weights_only=True)
    actual=torch.load(dense/'latents.pt',map_location='cpu',weights_only=True)
    equal=torch.equal(ref,actual);atomic(dense/'equivalence.json',dict(bitwise_equal=equal,reference=str(base/'dense/latents.pt')))
    assert equal,'Same-input Dense differs on new GPU'
    del ref,actual
    dest=base/a.method
    if dest.is_symlink():dest.unlink()
    dest.mkdir(exist_ok=True)
    atomic(dest/'paired_dense.json',dict(path=str(dense),gpu=gpu))
    args=['--summary','old' if a.method.startswith('old_') else 'new']
    if a.method.endswith('_risk'):args+=['--without-full-pset']
    run('local_experiments.video_runtime_v1.comparison_original',dest,args)
if __name__=='__main__':main()
