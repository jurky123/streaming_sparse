"""Dependency-aware GPU process scheduler, resumable without overwriting prior studies."""
import os,time,json,subprocess,itertools
from pathlib import Path
root=Path('/tmp/group_hierarchy_results');repo=Path('/tmp/group_response_178f95a8');src=Path(__file__).resolve().parents[1];cases=list(itertools.product(['horse_field','traffic_crossing'],[101,303]))
# G-K is admitted only after the inner-feature compatibility decision is recorded.
groups=['G-ORDER','G-KVR','G-K3-V2']
decision=root/'inner_decision.json'
if decision.exists():groups+=['G-K','G-K-joint','G-K3-V2-Kinner']
jobs=[dict(kind='restore',case=c,seed=s) for c,s in cases if not (root/'restoration'/f'{c}_s{s}.json').exists()]
for level in ['Dense','L3','L2_1','L2_8','L2_15','L2_22','L2_29']:
 for g in groups:
  for c,s in cases:
   if not (root/'video'/f'{c}_s{s}'/g/level/'result.json').exists():jobs.append(dict(kind='video',case=c,seed=s,group=g,level=level))
def folder(j):return root/'restoration'/f"{j['case']}_s{j['seed']}" if j['kind']=='restore' else root/'video'/f"{j['case']}_s{j['seed']}"/j['group']/j['level']
active={};failed=[]
while jobs or active:
 for gpu in range(8):
  if gpu in active:continue
  j=next((j for j in jobs if j['kind']=='restore' or j['level']=='Dense' or (folder(j).parent/'Dense/result.json').exists()),None)
  if j is None:continue
  jobs.remove(j);f=folder(j);f.mkdir(parents=True,exist_ok=True);log=(f/'worker.log').open('a');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(repo),OMP_NUM_THREADS='4')
  cmd=[str(repo/'streaming_sparse/.venv/bin/python'),'-u','-m','local_experiments.group_hierarchy.'+('restore' if j['kind']=='restore' else 'run')]
  for k,v in j.items():
   if k!='kind':cmd+=['--'+k,str(v)]
  p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT);active[gpu]=(p,log,j);print('START',gpu,j,flush=True)
 for gpu,(p,log,j) in list(active.items()):
  if p.poll() is None:continue
  log.close();code=p.returncode;del active[gpu];print('END',gpu,code,j,flush=True)
  if code:failed.append(dict(job=j,code=code))
 (root/'queue.json').write_text(json.dumps(dict(pending=jobs,active={str(g):j for g,(p,f,j) in active.items()},failed=failed),indent=2))
 if jobs and not active:break
 time.sleep(2)
subprocess.run(['rsync','-a','--exclude=qkv',str(root)+'/',str(src/'artifacts/group_hierarchy_178f95a8')+'/'],check=True)
if failed or jobs:raise SystemExit(1)
