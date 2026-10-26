import os,subprocess,json,time
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/opt11_finetune_20261025');out=root/'quality';out.mkdir(parents=True,exist_ok=True)
while not all((root/f'strong_training/layer{layer:02d}/completed.json').exists()for layer in range(1,30)):
 f=root/'strong_training/exit_codes.json'
 if f.exists()and any(json.loads(f.read_text())):raise RuntimeError('Training failed; quality not launched')
 time.sleep(5)
procs=[]
for rank in range(8):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR=f'/tmp/slimav_opt11_quality/rank{rank}/inductor',TRITON_CACHE_DIR=f'/tmp/slimav_opt11_quality/rank{rank}/triton');f=open(out/f'rank{rank}.log','w');p=subprocess.Popen(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt11.quality','--directory',str(root)],env=env,stdout=f,stderr=subprocess.STDOUT);procs.append((p,f))
codes=[p.wait()for p,_ in procs]
for _,f in procs:f.close()
(out/'exit_codes.json').write_text(json.dumps(codes));print(codes,flush=True);raise SystemExit(int(any(codes)))
