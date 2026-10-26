import json,os,subprocess,time
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024')
p=root/'optimization9/profile_Skip0Fast_step3/status.json';start=time.monotonic()
while not p.exists()or json.loads(p.read_text())['status']!='completed':
 if time.monotonic()-start>600:raise RuntimeError('Profile did not complete')
 time.sleep(3)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2')
subprocess.run(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt9.memory_probe','--directory',str(root)],env=env,check=True)
