import os,json,time,subprocess,sys
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024');start=time.monotonic()
while True:
 files=[root/f'optimization9/quality/memory_rank{r}.json'for r in range(8)]
 if all(p.exists()and json.loads(p.read_text())['status']=='completed'for p in files):break
 if time.monotonic()-start>900:raise RuntimeError('Memory quality not complete')
 time.sleep(5)
subprocess.run([sys.executable,'-u','-m','migration.unified_router.opt9.run_memory_benchmarks'],check=True)
