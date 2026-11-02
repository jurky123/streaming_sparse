"""Run matched one-step control after GPU4 end-to-end timings finish."""
import json,subprocess,sys,time
from pathlib import Path
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/sparse_expansion_20261031')
def main():
    lock=RunLock(ROOT/'horizon_control_queue')
    try:
        while True:
            s=json.loads((ROOT/'full_sparse_perf/status.json').read_text())
            if s['status']=='complete':break
            if s['status'] in ['failed','interrupted']:raise RuntimeError(s)
            atomic(ROOT/'horizon_control_queue/status.json',dict(status='waiting_for_gpu4_full_perf'));time.sleep(15)
        atomic(ROOT/'horizon_control_queue/status.json',dict(status='training_one_step_control'))
        subprocess.run([sys.executable,'-m','local_experiments.all_sparse_horizon_train','--horizon','1'],check=True)
        atomic(ROOT/'horizon_control_queue/status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'horizon_control_queue/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
