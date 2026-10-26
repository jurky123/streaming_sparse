import os,sys,subprocess,time,fcntl
from .setup import ROOT
from ..common import atomic_json

def main():
    out=ROOT/'capture';out.mkdir(exist_ok=True)
    lock=open(out/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[]
    for rank in range(8):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2')
        log=open(out/f'rank{rank}.log','a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt17.capture','--directory',str(ROOT)],env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((rank,p,log))
    atomic_json(out/'launcher.json',dict(pid=os.getpid(),started_at=time.time(),workers=[dict(rank=r,pid=p.pid) for r,p,_ in jobs]))
    codes={r:p.wait() for r,p,_ in jobs}
    for _,_,f in jobs:f.close()
    atomic_json(out/'exit_codes.json',codes)
    raise SystemExit(int(any(codes.values())))

if __name__=='__main__':main()
