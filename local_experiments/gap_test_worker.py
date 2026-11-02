import json,time,subprocess,sys
from pathlib import Path
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/four_gaps_20261031')
def main():
    lock=RunLock(ROOT/'test505_queue')
    try:
        while not (ROOT/'selection.json').exists():
            for p in [ROOT/'matched/status.json',ROOT/'shared/status.json',ROOT/'validation/status.json']:
                if p.exists() and json.loads(p.read_text())['status'] in ['failed','interrupted']:raise RuntimeError(p.read_text())
            time.sleep(15)
        atomic(ROOT/'test505_queue/status.json',dict(status='running'))
        subprocess.run([sys.executable,'-m','local_experiments.gap_quality','--phase','test','--seed','505'],check=True)
        atomic(ROOT/'test505_queue/status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'test505_queue/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
