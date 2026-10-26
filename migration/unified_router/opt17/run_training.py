import json,os,sys,subprocess,time
from pathlib import Path
from .setup import ROOT
from ..common import config,atomic_json,sha

def main():
    _,m=config(ROOT);out=ROOT/'training';out.mkdir(exist_ok=True)
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    assert json.loads((ROOT/'capture/exit_codes.json').read_text())==[0]*8
    for f,h in json.loads((ROOT/'parent_lock.json').read_text()).items():assert sha(f)==h,f
    atomic_json(ROOT/'source_manifest.json',{str(f.resolve()):sha(f)for f in Path(__file__).parent.glob('*.py')})
    assignments=[(layer,variant)for layer in m['layers']for variant in m['variants']];assert len(assignments)==8
    procs=[]
    for rank,(layer,variant)in enumerate(assignments):
        if (out/f'layer{layer:02d}/{variant}/completed.json').exists():continue
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',
            TORCHINDUCTOR_CACHE_DIR=f'/tmp/slimav_opt17/rank{rank}/inductor',TRITON_CACHE_DIR=f'/tmp/slimav_opt17/rank{rank}/triton')
        f=open(out/f'rank{rank}.log','a',buffering=1)
        proc=subprocess.Popen([sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt17.train','--layer',str(layer),'--variant',variant],env=env,stdout=f,stderr=subprocess.STDOUT)
        procs.append((rank,layer,variant,proc,f))
    atomic_json(out/'launcher.json',dict(pid=os.getpid(),started_at=time.time(),workers=[dict(rank=r,layer=l,variant=v,pid=p.pid)for r,l,v,p,_ in procs]))
    atomic_json(ROOT/'run_status.json',dict(status='running',workers=len(procs),started_at=time.time()))
    codes={r:p.wait()for r,_,_,p,_ in procs}
    for *_,f in procs:f.close()
    atomic_json(out/'exit_codes.json',codes)
    atomic_json(ROOT/'run_status.json',dict(status='completed'if not any(codes.values())else'failed',exit_codes=codes))
    raise SystemExit(int(any(codes.values())))

if __name__=='__main__':main()
