"""Finish independent jobs on idle GPUs, with a same-device dense calibration."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--jobs',nargs='+',required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args()
source=Path(__file__).resolve().parents[1];runtime=Path('/tmp/streaming_sparse-0/runtime');root=runtime/'artifacts/comparison_20261113';python=runtime/'streaming_sparse/.venv/bin/python'
env=os.environ.copy();env.update(PYTHONPATH=str(runtime),STREAM_EVAL_ROOT=str(root),STREAM_RESERVATION_PID=str(os.getpid()),CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',GAP_TAIL_MODE='1',TORCH_CUDA_ARCH_LIST='8.0',CUDA_HOME='/usr/local/cuda-12.2',MAX_JOBS='2')
if a.wait_pid:
 while Path(f'/proc/{a.wait_pid}/stat').exists():
  if Path(f'/proc/{a.wait_pid}/stat').read_text().split()[2]=='Z':break
  time.sleep(3)
failures=[]
def run(case,method,folder):
 if (folder/'result.json').exists():return
 folder.mkdir(parents=True,exist_ok=True)
 module='stream_eval_kernels' if method in ['pisa','solattn','svgear'] else 'stream_eval_official'
 cmd=[str(python),'-u','-m','local_experiments.'+module,'--method',method,'--inputs',str(root/case/'shared_inputs.pt'),'--output',str(folder)]
 (root/f'parallel.{a.gpu}.json').write_text(json.dumps(dict(status='running',gpu=a.gpu,case=case,method=method,pid=os.getpid())))
 with (folder/'worker.log').open('a') as log:subprocess.run(cmd,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for job in a.jobs:
 case,method=job.split('/')
 try:
  target=root/case/method
  if (target/'result.json').exists():continue
  device=root/'device_runs'/f'gpu{a.gpu}'/case
  reference=device/'dense'
  run(case,'dense',reference)
  # CPU-only check: paired baseline must reproduce the original Dense video exactly.
  import torch
  same=torch.equal(torch.load(reference/'latents.pt',map_location='cpu',weights_only=True),torch.load(root/case/'dense/latents.pt',map_location='cpu',weights_only=True))
  (reference/'cross_device_equivalence.json').write_text(json.dumps(dict(bitwise_equal=same,reference=str((root/case/'dense').relative_to(root)))))
  assert same,'Dense output differs across devices; inspect before comparison'
  folder=device/method;run(case,method,folder)
  d=json.loads((folder/'result.json').read_text());d['paired_dense']=str((reference/'result.json').relative_to(root));(folder/'result.json').write_text(json.dumps(d,indent=2))
  shutil.copytree(folder,target,dirs_exist_ok=True)
 except Exception as e:failures.append(dict(job=job,error=repr(e)))
 subprocess.run(['rsync','-a',str(root)+'/',str(source/'artifacts/comparison_20261113')+'/'],check=True)
(root/f'parallel.{a.gpu}.json').write_text(json.dumps(dict(status='finished',failures=failures)))
if failures:raise SystemExit(1)
