import os,json,time,subprocess
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/opt11_finetune_20261025')
while not(root/'paired/exit_codes.json').exists():
 for stage in ['strong_training','quality']:
  f=root/stage/'exit_codes.json'
  if f.exists()and any(json.loads(f.read_text())):raise RuntimeError(stage+' failed')
 time.sleep(5)
assert not any(json.loads((root/'paired/exit_codes.json').read_text()))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4')
with open(root/'report.log','w')as f:subprocess.run(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt11.report','--directory',str(root)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
