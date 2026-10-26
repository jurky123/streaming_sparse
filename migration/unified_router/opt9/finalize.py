import os,json,time,subprocess
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024');p=root/'optimization9/memory_paired/status.json';started=time.monotonic()
while not p.exists()or json.loads(p.read_text())['status']!='completed':
 if time.monotonic()-started>600:raise RuntimeError('Final paired timing did not complete')
 time.sleep(3)
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
for module in ['migration.unified_router.opt9.verify_outputs','migration.unified_router.opt9.report']:
 subprocess.run(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap',module,'--directory',str(root)],env=env,check=True)
