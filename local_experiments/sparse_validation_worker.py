import json,subprocess,sys,time
from pathlib import Path
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/sparse_expansion_20261031')
def main():
    lock=RunLock(ROOT/'validation_queue')
    try:
        while True:
            s=json.loads((ROOT/'followup/status.json').read_text())
            if s['status']=='complete':break
            if s['status']=='failed':raise RuntimeError(s)
            atomic(ROOT/'validation_queue/status.json',dict(status='waiting_for_followup_gpu4'));time.sleep(15)
        atomic(ROOT/'validation_queue/status.json',dict(status='running_validation'))
        subprocess.run([sys.executable,'-m','local_experiments.all_sparse_validate'],check=True)
        atomic(ROOT/'validation_queue/status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'validation_queue/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
