import subprocess,sys,os,json,fcntl
from .setup import ROOT,PARENT,TRAINING
from ..common import atomic_json,sha

def main():
 out=ROOT/'final';out.mkdir(exist_ok=True);lock=open(out/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic_json(out/'status.json',dict(status='running',pid=os.getpid()))
 atomic_json(ROOT/'source_manifest.json',{str(p.resolve()):sha(p) for p in __import__('pathlib').Path(__file__).parent.glob('*.py')})
 expected=json.loads((TRAINING/'chosen_checkpoints.json').read_text())
 for p,h in expected.items():assert sha(p)==h
 atomic_json(ROOT/'checkpoint_manifest.json',expected);jobs=[]
 for rank in range(8):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2')
  log=open(out/f'rank{rank}.log','a',buffering=1)
  p=subprocess.Popen([sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt22.evaluate','--case',str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((rank,p,log))
 codes={rank:p.wait() for rank,p,_ in jobs}
 for _,_,log in jobs:log.close()
 atomic_json(out/'exit_codes.json',codes);atomic_json(out/'status.json',dict(status='completed' if not any(codes.values()) else 'failed',exit_codes=codes))
if __name__=='__main__':main()
