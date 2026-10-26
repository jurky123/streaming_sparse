"""Continue unattended from training through final paired video evaluation."""
import fcntl,os,sys,subprocess,time,json
from .setup import ROOT
from ..common import atomic_json,sha

def main():
    lock=open(ROOT/'workflow.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    assert json.loads((ROOT/'graph_preflight.json').read_text())['passed']
    assert json.loads((ROOT/'initialize/completed.json').read_text())['success']
    atomic_json(ROOT/'workflow.json',dict(status='training_and_before_video',pid=os.getpid(),started_at=time.time()))
    code=subprocess.call([sys.executable,'-u','-m','migration.unified_router.opt20.run','--phase','training','--video-before'])
    if code:
        atomic_json(ROOT/'workflow.json',dict(status='training_failed',exit_code=code));raise SystemExit(code)
    assert len(list((ROOT/'training').glob('layer*/*/completed.json')))==58,'Training is incomplete'
    checkpoints={str(f.resolve()):sha(f) for f in (ROOT/'training').glob('layer*/*/best.pt')};assert len(checkpoints)==58
    atomic_json(ROOT/'chosen_checkpoints.json',checkpoints)
    atomic_json(ROOT/'workflow.json',dict(status='final_video',pid=os.getpid()))
    out=ROOT/'e2e_final';out.mkdir(exist_ok=True);jobs=[]
    for rank in range(8):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2')
        log=open(out/f'rank{rank}.log','a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt20.evaluate','--phase','final'],env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((rank,p,log))
    codes={rank:p.wait() for rank,p,_ in jobs}
    for _,_,log in jobs:log.close()
    atomic_json(out/'exit_codes.json',codes)
    if any(codes.values()):
        atomic_json(ROOT/'workflow.json',dict(status='final_video_failed',exit_codes=codes));raise SystemExit(1)
    code=subprocess.call([sys.executable,'-m','migration.unified_router.opt20.report'])
    atomic_json(ROOT/'workflow.json',dict(status='completed' if code==0 else 'report_failed',exit_code=code))
    raise SystemExit(code)

if __name__=='__main__':main()
