"""Run matched horizon RGB acceptance after both training arms complete."""
import json,subprocess,sys,time
from pathlib import Path
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/sparse_expansion_20261031')
def main():
    lock=RunLock(ROOT/'horizon_video_queue')
    try:
        while True:
            states=[]
            for h in [1,2]:
                p=ROOT/f'horizon{h}_train/status.json';s=json.loads(p.read_text()) if p.exists() else dict(status='initializing');states.append(s['status'])
                if s['status'] in ['failed','interrupted']:raise RuntimeError(dict(horizon=h,state=s))
            if all(s=='pilot_training_complete_pending_validation' for s in states):break
            atomic(ROOT/'horizon_video_queue/status.json',dict(status='waiting_for_both_horizons',arms=states));time.sleep(15)
        atomic(ROOT/'horizon_video_queue/status.json',dict(status='running_rgb_acceptance'))
        subprocess.run([sys.executable,'-m','local_experiments.all_sparse_horizon_video'],check=True)
        atomic(ROOT/'horizon_video_queue/status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'horizon_video_queue/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
