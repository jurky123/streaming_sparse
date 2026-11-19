"""Sequential same-device comparison; all failures retained and reported."""
import os,sys,subprocess,json,time
from pathlib import Path
ROOT=Path('/tmp/streaming_sparse-0/runtime');OUT=ROOT/'artifacts/comparison_20261113'
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),STREAM_EVAL_ROOT=str(OUT),STREAM_RESERVATION_PID=str(os.getpid()),CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES','0'),OMP_NUM_THREADS='4',GAP_TAIL_MODE='1')
while True:
 try:os.kill(813148,0)
 except ProcessLookupError:break
 time.sleep(5)
rows=[]
for case in ['horse_field','traffic_crossing']:
 for method in ['pisa','solattn']:
  out=OUT/case/method;out.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-u','-m','local_experiments.stream_eval_kernels','--inputs',str(OUT/case/'shared_inputs.pt'),'--output',str(out)]
  if method!='ours':cmd+=['--method',method]
  if (out/'result.json').exists():continue
  (OUT/'status.json').write_text(json.dumps(dict(case=case,method=method,status='running',pid=os.getpid())))
  with (out/'worker.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  rows.append(dict(case=case,method=method,exit_code=r.returncode))
  (OUT/'kernel_execution.json').write_text(json.dumps(rows,indent=2))
(OUT/'status.json').write_text(json.dumps(dict(status='finished',rows=rows)))
