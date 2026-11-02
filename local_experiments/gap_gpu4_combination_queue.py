"""Wait for our previous GPU4 queue, then validate composition; no external locks."""
import os,time,json,fcntl
from pathlib import Path
from . import gap_gpu4_queue as queue
ROOT=Path('artifacts/gap_gpu4_combination_queue_20261102')
def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'followup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (ROOT/'waiting.json').write_text(json.dumps(dict(pid=os.getpid(),waiting_for_pid=3048127))+'\n')
        while True:
            try:
                cmd=Path('/proc/3048127/cmdline').read_bytes()
                live=b'local_experiments.gap_gpu4_queue' in cmd and Path('/proc/3048127/stat').read_text().split(') ',1)[1][0]!='Z'
            except FileNotFoundError:live=False
            if not live:break
            time.sleep(5)
        os.environ['GAP_WORKING_GIB']='24'
        queue.ROOT=ROOT
        queue.JOBS=[('gap_staged_history_quality','/tmp/gap_staged_history_quality_20261102.log'),('gap_staged_history_perf','/tmp/gap_staged_history_perf_20261102.log')]
        queue.main()
if __name__=='__main__':main()
