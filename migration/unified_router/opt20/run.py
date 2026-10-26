"""Eight GPU slots, persistent dynamic queue, with per-job exit records."""
import argparse,concurrent.futures,fcntl,os,queue,subprocess,sys,time,json,threading
from .setup import ROOT
from ..common import config,atomic_json,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initialize','training'],required=True)
    p.add_argument('--video-before',action='store_true');a=p.parse_args()
    out=ROOT/a.phase;out.mkdir(exist_ok=True)
    lock=open(out/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    _,m=config(ROOT);layers=[24,1,14,29]+[l for l in m['layers'] if l not in [24,1,14,29]]
    assert all(c==0 for c in json.loads((ROOT/'capture/exit_codes.json').read_text()).values())
    if a.phase=='training':assert json.loads((ROOT/'preflight.json').read_text())['passed']
    tasks=queue.Queue()
    for layer in layers:
        for variant in (m['variants'] if a.phase=='training' else [None]):
            done=ROOT/f'training/layer{layer:02d}/{variant}/completed.json' if variant else ROOT/f'initialization/layer{layer:02d}.pt'
            if not done.exists():tasks.put((layer,variant))
    total=tasks.qsize();results={};mutex=threading.Lock()
    def worker(rank):
        if rank==0 and a.video_before:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2')
            video_dir=ROOT/'e2e_before';video_dir.mkdir(exist_ok=True)
            existing=os.environ.get('OPT20_BEFORE_PID')
            if existing:
                from pathlib import Path
                procfile=Path('/proc')/existing/'stat'
                atomic_json(out/f'worker{rank}.json',dict(status='video_before',pid=int(existing),adopted=True))
                while procfile.exists():
                    try:state=procfile.read_text().rsplit(')',1)[1].split()[0]
                    except FileNotFoundError:break
                    if state=='Z':break
                    time.sleep(2)
            code=0
            if len(list(video_dir.glob('*.seed0/completed.json')))!=8:
                with (video_dir/'worker.log').open('a',buffering=1) as log:
                    proc=subprocess.Popen([sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt20.evaluate','--phase','before'],env=env,stdout=log,stderr=subprocess.STDOUT)
                    atomic_json(out/f'worker{rank}.json',dict(status='video_before',pid=proc.pid))
                    code=proc.wait()
            atomic_json(video_dir/'exit_code.json',dict(exit_code=code))
        while True:
            try:layer,variant=tasks.get_nowait()
            except queue.Empty:return
            key=f'layer{layer:02d}'+('_'+variant if variant else '')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2')
            cmd=[sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt20.'+('train' if variant else 'initialize'),'--layer',str(layer)]
            if variant:cmd+=['--variant',variant]
            with (out/(key+'.log')).open('a',buffering=1) as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(out/f'worker{rank}.json',dict(status='running',layer=layer,variant=variant,pid=proc.pid,started_at=time.time()))
                code=proc.wait()
            with mutex:
                results[key]=code;atomic_json(out/'exit_codes.json',results)
                atomic_json(out/'queue.json',dict(total=total,finished=len(results),failed=sum(c!=0 for c in results.values()),remaining=tasks.qsize()))
            atomic_json(out/f'worker{rank}.json',dict(status='idle' if code==0 else 'job_failed',layer=layer,variant=variant,exit_code=code))
    atomic_json(out/'launcher.json',dict(pid=os.getpid(),started_at=time.time(),tasks=total,gpus=8))
    atomic_json(out/'exit_codes.json',{})
    atomic_json(out/'queue.json',dict(total=total,finished=0,failed=0,remaining=total))
    for rank in range(8):atomic_json(out/f'worker{rank}.json',dict(status='pending'))
    atomic_json(ROOT/(a.phase+'_source_manifest.json'),{str(f.resolve()):sha(f) for f in __import__('pathlib').Path(__file__).parent.glob('*.py')})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
    atomic_json(out/'completed.json',dict(success=not any(results.values()),jobs=results))
    raise SystemExit(int(any(results.values())))

if __name__=='__main__':main()
