import os,subprocess,sys
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024');procs=[]
for rank in range(8):
 out=root/('optimization9/paired'if rank==0 else f'optimization9/paired_val/prompt{rank}');out.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',RANK=str(rank))
 cmd=['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt9.paired_bench','--directory',str(root),'--prompt-index',str(rank),'--repeats','6'if rank==0 else'3']
 if rank:cmd.append('--two')
 f=open(out/'run.log','w');p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);procs.append((p,f))
codes=[p.wait()for p,_ in procs]
for _,f in procs:f.close()
print('exit_codes',codes,flush=True);sys.exit(int(any(codes)))
