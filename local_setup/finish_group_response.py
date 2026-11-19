"""Finish the fixed experiment queue, retry failed units, then validate and report."""
import json,time,subprocess,os,shutil
from pathlib import Path
source=Path(__file__).resolve().parents[1];root=Path('/tmp/group_response_results');repo=Path('/tmp/group_response_178f95a8');py=str(repo/'streaming_sparse/.venv/bin/python')
while True:
 try:q=json.loads((root/'queue.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
 if not q['pending'] and not q['active']:break
 time.sleep(5)
# Controller has finished all GPU work; its final sync is read-only with respect to raw outputs.
time.sleep(3)
for p in (source/'local_experiments/group_response').glob('*.py'):shutil.copy2(p,repo/'local_experiments/group_response'/p.name)
missing=[(c,s,g,l) for c in ['horse_field','traffic_crossing'] for s in [101,303] for g in ['G-K','G-VR','G-KVR','G-ORDER'] for l in ['Dense','L2','L3'] if not (root/'video'/f'{c}_s{s}'/g/l/'result.json').exists()]
retry=None
if missing:
 log=open('/tmp/group_response_final_retry.log','w');retry=subprocess.Popen(['python',str(source/'local_setup/run_group_response.py')],cwd=source,stdout=log,stderr=subprocess.STDOUT)
# One known failed run uses GPU0; avoid sharing any GPU if more retry work appeared.
if len(missing)>7:assert retry.wait()==0;retry=None
cost_gpu=7 if len(missing)<=7 else 0
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(cost_gpu),PYTHONPATH=str(repo),OMP_NUM_THREADS='4')
with open('/tmp/group_response_paired_cost.log','w') as f:subprocess.run([py,'-u','-m','local_experiments.group_response.cost'],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
if retry is not None:assert retry.wait()==0
subprocess.run(['rsync','-a',str(root)+'/',str(source/'artifacts/group_response_178f95a8')+'/'],check=True)
for script in ['report_group_response.py','audit_group_response.py','visualize_group_response.py']:
 subprocess.run([py,str(source/'local_setup'/script)],cwd=source,check=True)
print('FROZEN EXPERIMENT AND AUDIT COMPLETE',flush=True)
