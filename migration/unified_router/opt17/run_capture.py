import os,sys,time,subprocess,json
from .setup import ROOT
from ..common import atomic_json

def main():
    out=ROOT/'capture';out.mkdir(parents=True,exist_ok=True);procs=[]
    for rank in range(8):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2')
        f=open(out/f'rank{rank}.log','a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt17.capture','--directory',str(ROOT)],env=env,stdout=f,stderr=subprocess.STDOUT)
        procs.append((rank,p,f))
    atomic_json(out/'launcher.json',dict(pid=os.getpid(),started_at=time.time(),workers=[dict(rank=r,pid=p.pid)for r,p,_ in procs]))
    codes=[p.wait()for _,p,_ in procs]
    for _,_,f in procs:f.close()
    atomic_json(out/'exit_codes.json',codes)
    raise SystemExit(int(any(codes)))

if __name__=='__main__':main()
