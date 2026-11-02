import json,subprocess,sys,time
from pathlib import Path
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/four_gaps_20261031')
def main():
    lock=RunLock(ROOT/'validation_queue')
    try:
        while not (ROOT/'graph_checks.json').exists():time.sleep(10)
        assert json.loads((ROOT/'graph_checks.json').read_text())['passed']
        atomic(ROOT/'validation_queue/status.json',dict(status='validation_then_fresh_seed404'))
        subprocess.run([sys.executable,'-m','local_experiments.gap_quality','--phase','val'],check=True)
        subprocess.run([sys.executable,'-m','local_experiments.gap_quality','--phase','test','--seed','404'],check=True)
        atomic(ROOT/'validation_queue/status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'validation_queue/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
