"""Resume follow-ups after each assigned GPU's paired benchmark; one job per GPU."""
import os,json,time,subprocess,sys
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024');out=root/'optimization10';active={};pending=set([0,1,2,3,4,5,7]);done=[];micro_started=False
def completed(f):return f.exists()and json.loads(f.read_text()).get('status')=='completed'
def run(gpu,module,args,log,extra=None):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',**(extra or{}));f=open(log,'w');p=subprocess.Popen(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt10.'+module,'--directory',str(root)]+args,stdout=f,stderr=subprocess.STDOUT,env=env);return p,f
while pending or active or not micro_started:
 for gpu in list(pending):
  if 1<=gpu<=5:
   variant=['Opt9','Wide','Head','Moments','Compact'][gpu-1];finished=out/f'profile_{variant}_step3/status.json'
  else:rank=0 if gpu==0 else 1;finished=out/f'changed_audit/rank{rank}.json'
  if completed(finished):pending.remove(gpu);continue
  bench=out/('paired/status.json'if gpu==0 else f'paired_val/prompt{gpu}/status.json')
  if not completed(bench):continue
  if 1<=gpu<=5:job=run(gpu,'profile',['--variant',variant],out/f'profile_{variant}.log')
  else:job=run(gpu,'changed_audit',[],out/f'changed_audit_rank{rank}.log',dict(RANK=str(rank),WORLD_SIZE='2'))
  active[gpu]=job;pending.remove(gpu)
 for gpu,(p,f)in list(active.items()):
  code=p.poll()
  if code is not None:f.close();done.append(code);del active[gpu];print('gpu',gpu,'exit',code,flush=True)
 if 7 not in pending and 7 not in active and not micro_started:
  if any(done):sys.exit(1)
  active[7]=run(7,'micro_suite',[],out/'micro_suite.log');micro_started=True
 time.sleep(5)
sys.exit(int(any(done)))
