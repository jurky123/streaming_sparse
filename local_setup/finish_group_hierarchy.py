"""Complete added factorial conditions and local evaluation after the first GPU wave."""
import os,time,json,subprocess,shutil
from pathlib import Path
src=Path(__file__).resolve().parents[1];root=Path('/tmp/group_hierarchy_results');repo=Path('/tmp/group_response_178f95a8');py=str(repo/'streaming_sparse/.venv/bin/python')
while True:
 try:q=json.loads((root/'queue.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
 if not q['pending'] and not q['active']:break
 time.sleep(5)
time.sleep(3)
for p in (src/'local_experiments/group_hierarchy').glob('*.py'):shutil.copy2(p,repo/'local_experiments/group_hierarchy'/p.name)
def launch(module,args,gpu,log):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(repo),OMP_NUM_THREADS='4');f=open(log,'w');return subprocess.Popen([py,'-u','-m','local_experiments.group_hierarchy.'+module]+args,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
p=launch('preflight',[],0,'/tmp/group_hierarchy_preflight_final.log');assert p.wait()==0
jobs=[]
for gpu,(c,s) in enumerate([(c,s) for c in ['horse_field','traffic_crossing'] for s in [101,303]]):jobs.append(launch('evaluate_saved',['--case',c,'--seed',str(s)],gpu,f'/tmp/group_hierarchy_local_{c}_{s}.log'))
jobs.append(launch('cost',[],7,'/tmp/group_hierarchy_cost.log'))
assert all(p.wait()==0 for p in jobs)
with open('/tmp/group_hierarchy_queue_phase2.log','w') as f:subprocess.run(['python',str(src/'local_setup/run_group_hierarchy.py')],cwd=src,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['rsync','-a',str(root)+'/',str(src/'artifacts/group_hierarchy_178f95a8')+'/'],check=True)
print('ALL SIX-CONDITION RUNS COMPLETE',flush=True)
