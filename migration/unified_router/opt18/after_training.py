"""Start the measured sweep only after all continuation workers exit cleanly."""
import fcntl,json,os,subprocess,sys,time
from .setup import ROOT
from ..common import atomic_json

def main():
    lock=open(ROOT/'frontier_queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(ROOT/'frontier_queue.json',dict(pid=os.getpid(),status='waiting_for_training'))
    path=ROOT/'training/exit_codes.json'
    while not path.exists():time.sleep(5)
    codes=json.loads(path.read_text())
    if any(codes.values()):
        atomic_json(ROOT/'frontier_queue.json',dict(pid=os.getpid(),status='blocked_by_training_failure',exit_codes=codes));raise SystemExit(1)
    atomic_json(ROOT/'frontier_queue.json',dict(pid=os.getpid(),status='running_frontier'))
    code=subprocess.call([sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt18.run','--phase','frontier'])
    atomic_json(ROOT/'frontier_queue.json',dict(pid=os.getpid(),status='completed'if code==0 else'failed',exit_code=code))
    raise SystemExit(code)

if __name__=='__main__':main()
