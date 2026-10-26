"""Persistent supervisors for continuation and subsequent measured frontier."""
import argparse,fcntl,json,os,sys,subprocess,time
from pathlib import Path
from .setup import ROOT
from ..common import config,atomic_json,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['training','frontier'],default='training');a=p.parse_args()
    _,m=config(ROOT);out=ROOT/a.phase;out.mkdir(exist_ok=True)
    lockfile=open(out/'supervisor.lock','w');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    assert json.loads((Path(m['parent_opt17'])/'capture/exit_codes.json').read_text())==[0]*8
    for f,h in json.loads((ROOT/'parent_lock.json').read_text()).items():assert sha(f)==h,f
    assignments=[(layer,variant)for layer in m['layers']for variant in m['variants']]
    if a.phase=='frontier':
        assert all((ROOT/f'training/layer{l:02d}/{v}/completed.json').exists()for l,v in assignments)
    atomic_json(ROOT/(a.phase+'_source_manifest.json'),{str(f.resolve()):sha(f)for f in Path(__file__).parent.glob('*.py')})
    procs=[]
    for rank,(layer,variant)in enumerate(assignments):
        if (out/f'layer{layer:02d}/{variant}/completed.json').exists():continue
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR=f'/tmp/slimav_opt18/rank{rank}/inductor',TRITON_CACHE_DIR=f'/tmp/slimav_opt18/rank{rank}/triton')
        f=open(out/f'rank{rank}.log','a',buffering=1)
        module='train'if a.phase=='training'else'frontier'
        proc=subprocess.Popen([sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt18.'+module,'--layer',str(layer),'--variant',variant],env=env,stdout=f,stderr=subprocess.STDOUT)
        procs.append((rank,layer,variant,proc,f))
    atomic_json(out/'launcher.json',dict(pid=os.getpid(),started_at=time.time(),phase=a.phase,workers=[dict(rank=r,layer=l,variant=v,pid=p.pid)for r,l,v,p,_ in procs]))
    atomic_json(ROOT/'run_status.json',dict(status=a.phase,workers=len(procs),started_at=time.time()))
    codes={r:p.wait()for r,_,_,p,_ in procs}
    for *_,f in procs:f.close()
    atomic_json(out/'exit_codes.json',codes)
    atomic_json(ROOT/'run_status.json',dict(status=a.phase+'_completed'if not any(codes.values())else a.phase+'_failed',exit_codes=codes))
    raise SystemExit(int(any(codes.values())))

if __name__=='__main__':main()
