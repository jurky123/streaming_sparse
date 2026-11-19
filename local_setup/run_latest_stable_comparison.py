"""Run latest algorithm with efficient execution on the original scene GPUs."""
import os,subprocess,json,shutil
from pathlib import Path
source=Path(__file__).resolve().parents[1];root=Path('/tmp/streaming_sparse-0/runtime');out=root/'artifacts/comparison_20261113';jobs=[]
for relative in ['local_experiments/stream_eval_latest_stable.py','local_experiments/gap_exact_round_kernel.py','local_experiments/review_v1/ops.py','local_experiments/review_v1/ablation.py','local_experiments/review_v1/fast_attention.py','local_experiments/review_v1/fast_route.py','local_experiments/review_v1/fast_runtime.py','local_experiments/review_v1/fused_route.py','local_experiments/review_v1/fused_runtime.py','local_experiments/review_v1/optimized_runtime.py','local_experiments/review_v1/graph_summary.py','local_experiments/review_v1/tuned_attention.py']:
 shutil.copy2(source/relative,root/relative)
for gpu,case in [('0','horse_field'),('1','traffic_crossing')]:
 folder=out/case/'ours_latest_stable';folder.mkdir(exist_ok=True)
 if (folder/'result.json').exists():continue
 env=os.environ.copy();env.update(PYTHONPATH=str(root),STREAM_EVAL_ROOT=str(out),STREAM_RESERVATION_PID=str(os.getpid()),CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',GAP_TAIL_MODE='1')
 log=(folder/'worker.log').open('a')
 cmd=[str(root/'streaming_sparse/.venv/bin/python'),'-u','-m','local_experiments.stream_eval_latest_stable','--inputs',str(out/case/'shared_inputs.pt'),'--output',str(folder)]
 jobs.append((case,subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT),log))
failures=[]
for case,job,log in jobs:
 code=job.wait();log.close();print(case,code,flush=True)
 if code:failures.append(dict(case=case,exit_code=code))
 subprocess.run(['rsync','-a',str(out)+'/',str(source/'artifacts/comparison_20261113')+'/'],check=True)
(source/'artifacts/comparison_20261113/latest_stable_failures.json').write_text(json.dumps(failures))
if failures:raise SystemExit(1)
