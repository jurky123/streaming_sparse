import os,subprocess,json
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/opt11_finetune_20261025');(root/'capture').mkdir(parents=True,exist_ok=True);procs=[]
for rank in range(8):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2');f=open(root/f'capture/rank{rank}.log','w');p=subprocess.Popen(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt11.capture','--directory',str(root)],env=env,stdout=f,stderr=subprocess.STDOUT);procs.append((p,f))
codes=[p.wait()for p,_ in procs]
for _,f in procs:f.close()
(root/'capture/exit_codes.json').write_text(json.dumps(codes));print(codes,flush=True);raise SystemExit(int(any(codes)))
