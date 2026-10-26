import os,subprocess,json,time
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/opt11_finetune_20261025');out=root/'training';out.mkdir(parents=True,exist_ok=True)
while not(root/'preflight.json').exists():time.sleep(3)
assert json.loads((root/'preflight.json').read_text())['passed']
while not all(json.loads((root/f'capture/rank{i}.json').read_text())['status']=='completed'for i in range(8)):time.sleep(3)
procs=[]
for rank in range(8):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR=f'/tmp/slimav_opt11/rank{rank}/inductor',TRITON_CACHE_DIR=f'/tmp/slimav_opt11/rank{rank}/triton');f=open(out/f'rank{rank}.log','w');p=subprocess.Popen(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt11.train','--directory',str(root)],env=env,stdout=f,stderr=subprocess.STDOUT);procs.append((p,f))
codes=[p.wait()for p,_ in procs]
for _,f in procs:f.close()
(out/'exit_codes.json').write_text(json.dumps(codes));print(codes,flush=True);raise SystemExit(int(any(codes)))
