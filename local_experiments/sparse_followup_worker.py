"""Wait for this experiment's training, then run non-overlapping GPU audits."""
import json,subprocess,sys,time
from pathlib import Path
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/sparse_expansion_20261031')
def main():
    lock=RunLock(ROOT/'followup')
    try:
        while True:
            s=json.loads((ROOT/'all_sparse_train/status.json').read_text())
            if s['status']=='pilot_training_complete_pending_validation':break
            if s['status'] in ['failed','interrupted']:raise RuntimeError(s)
            atomic(ROOT/'followup/status.json',dict(status='waiting_for_training_gpu4'));time.sleep(15)
        for module in ['sparse_refine_bench','all_sparse_twostep_smoke']:
            atomic(ROOT/'followup/status.json',dict(status='running',module=module))
            subprocess.run([sys.executable,'-m','local_experiments.'+module],check=True)
        atomic(ROOT/'followup/status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'followup/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
