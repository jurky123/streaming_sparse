import os,subprocess,time,json,sys
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024');procs=[]
# Wait for all extra-prompt timing GPUs, keeping profiling out of those timings.
for rank in range(1,8):
 path=root/f'optimization9/paired_val/prompt{rank}/status.json'
 started=time.monotonic()
 while not path.exists()or json.loads(path.read_text())['status']!='completed':
  if time.monotonic()-started>1200:raise RuntimeError(f'Benchmark GPU {rank} did not complete')
  time.sleep(5)
for gpu,variant,step in [(1,'Skip2',3),(2,'Skip0',3),(3,'Skip2Fast',3),(4,'Skip2Fast',0),(5,'Opt8Full',3),(6,'Packed',3),(7,'Skip0Fast',3)]:
 out=root/f'optimization9/profile_{variant}_step{step}';out.mkdir(exist_ok=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',SLIMAV_PROFILE_STEP=str(step))
 f=open(out/'run.log','w');p=subprocess.Popen(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt9.profile','--directory',str(root),'--variant',variant],env=env,stdout=f,stderr=subprocess.STDOUT);procs.append((p,f))
codes=[p.wait()for p,_ in procs]
for _,f in procs:f.close()
print('exit_codes',codes,flush=True);sys.exit(int(any(codes)))
