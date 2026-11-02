"""GPU5: finish family ablations, then second test seed, publish partial report."""
import json,subprocess,sys,time
from pathlib import Path
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/sparse_quality_line_20261031')
def main():
    lock=RunLock(ROOT/'queue')
    try:
        while True:
            subprocess.run([sys.executable,'-m','local_experiments.sparse_quality_report'],check=True)
            p=ROOT/'ablation/status.json';s=json.loads(p.read_text()) if p.exists() else dict(status='initializing')
            if s['status']=='complete':break
            if s['status'] in ['failed','interrupted']:raise RuntimeError(s)
            atomic(ROOT/'queue/status.json',dict(status='waiting_for_ablation_gpu5'));time.sleep(30)
        atomic(ROOT/'queue/status.json',dict(status='broad_seed303'))
        subprocess.run([sys.executable,'-m','local_experiments.sparse_quality_line','--mode','broad','--seed','303'],check=True)
        while True:
            subprocess.run([sys.executable,'-m','local_experiments.sparse_quality_report'],check=True)
            s=json.loads((ROOT/'broad_s202/status.json').read_text())
            if s['status']=='complete':break
            if s['status'] in ['failed','interrupted']:raise RuntimeError(s)
            time.sleep(30)
        atomic(ROOT/'queue/status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'queue/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
