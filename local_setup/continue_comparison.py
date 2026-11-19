"""Resume missing benchmark results on one idle physical GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path('/tmp/streaming_sparse-0/runtime'));p.add_argument('--gpu',default='0');p.add_argument('--cases',nargs='+',default=['horse_field','traffic_crossing']);p.add_argument('--methods',nargs='+',default=['dense','lightforcing','dasa','forcingkv','ours','pisa','solattn','svgear']);p.add_argument('--wait-pid',type=int);a=p.parse_args()
source=Path(__file__).resolve().parents[1];root=a.runtime.resolve();out=root/'artifacts/comparison_20261113';python=root/'streaming_sparse/.venv/bin/python'
env=os.environ.copy();env.update(PYTHONPATH=str(root),STREAM_EVAL_ROOT=str(out),STREAM_RESERVATION_PID=str(os.getpid()),CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',GAP_TAIL_MODE='1',TORCH_CUDA_ARCH_LIST='8.0',CUDA_HOME='/usr/local/cuda-12.2',MAX_JOBS='2')
if a.wait_pid:
 while Path(f'/proc/{a.wait_pid}/stat').exists():
  if Path(f'/proc/{a.wait_pid}/stat').read_text().split()[2]=='Z':break
  time.sleep(5)
failures=[]
for case in a.cases:
 for method in a.methods:
  folder=out/case/method;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'result.json').exists():continue
  if method!='dense' and not (out/case/'dense/result.json').exists():
   failures.append(dict(case=case,method=method,error='dense baseline missing'));continue
  module='stream_eval_ours' if method=='ours' else 'stream_eval_kernels' if method in ['pisa','solattn','svgear'] else 'stream_eval_official'
  cmd=[str(python),'-u','-m','local_experiments.'+module,'--inputs',str(out/case/'shared_inputs.pt'),'--output',str(folder)]
  if method!='ours':cmd+=['--method',method]
  if (folder/'worker.log').exists():(folder/'worker.log').rename(folder/f'worker.{time.time_ns()}.log')
  (out/f'controller.{a.gpu}.json').write_text(json.dumps(dict(status='running',case=case,method=method,gpu=a.gpu,pid=os.getpid())))
  with (folder/'worker.log').open('w') as log:result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:failures.append(dict(case=case,method=method,exit_code=result.returncode))
  subprocess.run(['rsync','-a',str(out)+'/',str(source/'artifacts/comparison_20261113')+'/'],check=True)
(out/f'resume_failures.{a.gpu}.json').write_text(json.dumps(failures,indent=2))
(out/f'controller.{a.gpu}.json').write_text(json.dumps(dict(status='finished',failures=failures)))
subprocess.run([sys.executable,str(source/'local_setup/report_comparison.py')],check=True)
if failures:raise SystemExit(1)
