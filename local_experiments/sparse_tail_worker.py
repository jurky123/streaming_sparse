import json,subprocess,sys,time
from pathlib import Path
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/sparse_expansion_20261031')
def main():
    lock=RunLock(ROOT/'tail_queue')
    try:
        while True:
            p=ROOT/'horizon2_train/status.json';s=json.loads(p.read_text()) if p.exists() else dict(status='initializing')
            if s['status']=='pilot_training_complete_pending_validation':break
            if s['status'] in ['failed','interrupted']:raise RuntimeError(s)
            atomic(ROOT/'tail_queue/status.json',dict(status='waiting_for_gpu5_horizon2'));time.sleep(15)
        atomic(ROOT/'tail_queue/status.json',dict(status='tuning_conditional_tail'))
        subprocess.run([sys.executable,'-m','local_experiments.conditional_v_tune'],check=True)
        atomic(ROOT/'tail_queue/status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'tail_queue/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
