"""Run one Val prompt per GPU; all eight exit statuses must succeed."""
import os,subprocess,sys
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024')
processes=[]
for rank in range(8):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',RANK=str(rank),WORLD_SIZE='8')
 log=open(root/f'optimization9/quality/memory_rank{rank}.log','w')
 proc=subprocess.Popen(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt9.memory_quality','--directory',str(root),'--allow-quant-error','--resume'],env=env,stdout=log,stderr=subprocess.STDOUT)
 processes.append((proc,log))
codes=[p.wait()for p,_ in processes]
for _,f in processes:f.close()
print('exit_codes',codes,flush=True)
sys.exit(int(any(codes)))
