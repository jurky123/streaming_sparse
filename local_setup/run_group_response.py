"""One isolated subprocess per GPU, dependency-aware experiment queue (not agents)."""
import os,time,json,subprocess,itertools,shutil
from pathlib import Path
root=Path('/tmp/group_response_results');repo=Path('/tmp/group_response_178f95a8');source=Path(__file__).resolve().parents[1];root.mkdir(exist_ok=True)
cases=list(itertools.product(['horse_field','traffic_crossing'],[101,303]));groups=['G-KVR','G-K','G-VR','G-ORDER']
jobs=[dict(case=c,seed=s,group=g,level=l) for l in ['Dense','L3','L2'] for g in groups for c,s in cases]
def folder(j):return root/'video'/f"{j['case']}_s{j['seed']}"/j['group']/j['level']
jobs=[j for j in jobs if not (folder(j)/'result.json').exists()];active={};failed=[]
while jobs or active:
 for gpu in range(8):
  if gpu in active:continue
  ready=next((j for j in jobs if j['level']=='Dense' or (folder(j).parent/'Dense/result.json').exists()),None)
  if ready is None:continue
  jobs.remove(ready);p=folder(ready);p.mkdir(parents=True,exist_ok=True);log=(p/'worker.log').open('a');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(repo),OMP_NUM_THREADS='4')
  args=[str(repo/'streaming_sparse/.venv/bin/python'),'-u','-m','local_experiments.group_response.run']
  for k,v in ready.items():args+=['--'+k,str(v)]
  if ready['level']=='L3' and ready['group']=='G-KVR':args+=['--collect']
  process=subprocess.Popen(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT);active[gpu]=(process,log,ready);print('START',gpu,ready,flush=True)
 for gpu,(p,log,j) in list(active.items()):
  code=p.poll()
  if code is None:continue
  log.close();del active[gpu];print('END',gpu,code,j,flush=True)
  if code:failed.append(dict(job=j,code=code))
 (root/'queue.json').write_text(json.dumps(dict(pending=jobs,active={str(g):j for g,(p,f,j) in active.items()},failed=failed),indent=2))
 if jobs and not active:break
 time.sleep(2)
subprocess.run(['rsync','-a',str(root)+'/',str(source/'artifacts/group_response_178f95a8')+'/'],check=True)
if failed or jobs:raise SystemExit(1)
print('ALL 48 VIDEO RUNS COMPLETE',flush=True)
