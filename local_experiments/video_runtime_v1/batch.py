"""One resident process per GPU; generation finishes before decode/quality stage."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .queue import Queue
from .deployment import CURRENT
from ..learned_cache.io import atomic,RunLock

def alive(pid):
    proc=Path('/proc')/str(pid)
    try:return proc.exists() and proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]!='Z' and b'local_experiments.video_runtime_v1.worker' in proc.joinpath('cmdline').read_bytes()
    except (FileNotFoundError,ProcessLookupError):return False

def worker_command(args,stage):
    command=[sys.executable,'-u','-m','local_experiments.video_runtime_v1.worker','--root',str(args.root),'--stage',stage,'--tile-batch',str(args.tile_batch),'--model-seed',str(args.model_seed),'--history-pages',str(args.history_pages)]
    for name in ('checkpoint','reader_checkpoint','local_assets'):
        value=getattr(args,name)
        if value:command.extend(['--'+name.replace('_','-'),str(value)])
    if args.profile:command.append('--profile')
    if args.validate:command.append('--validate')
    return command

def run(args):
    root=Path(args.root);lock=RunLock(root/'batch');q=Queue(root);q.initialize(args.manifest);q.recover()
    try:
        for stage in ('generate','postprocess'):
            if all(r['status']=='done' for r in q.status() if r['stage']==stage):continue
            children=[]
            for gpu in args.gpus:
                record=root/'batch'/f'{stage}_gpu{gpu}.json'
                previous=json.loads(record.read_text()) if record.exists() else {}
                if alive(previous.get('pid')):
                    children.append((gpu,None,previous['pid']));continue
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
                command=worker_command(args,stage)
                path=root/'batch'/f'{stage}_gpu{gpu}.log'
                with path.open('a') as log:p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append((gpu,p,p.pid));atomic(root/'batch'/f'{stage}_gpu{gpu}.json',dict(pid=p.pid,command=command))
            while any(p.poll() is None if p is not None else alive(pid) for _,p,pid in children):
                atomic(root/'batch/status.json',dict(stage=stage,queue=q.status(),updated=time.time()))
                if any(p is not None and p.poll() not in (None,0) for _,p,_ in children):raise RuntimeError(f'{stage} worker failed; inspect batch/{stage}_gpu*.log')
                time.sleep(10)
            if any(p is not None and p.returncode for _,p,_ in children):raise RuntimeError(f'{stage} worker failed')
            status=q.status()
            if any(r['stage']==stage and r['status']!='done' for r in status):raise RuntimeError(f'Incomplete {stage}: {status}')
        atomic(root/'batch/status.json',dict(stage='complete',queue=q.status(),updated=time.time()))
    except BaseException as exc:
        atomic(root/'batch/status.json',dict(stage='failed',error=repr(exc),queue=q.status(),updated=time.time()));raise
    finally:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--gpus',type=int,nargs='+',required=True);p.add_argument('--tile-batch',type=int,default=CURRENT['tile_batch']);p.add_argument('--model-seed',type=int,default=17);p.add_argument('--checkpoint');p.add_argument('--reader-checkpoint');p.add_argument('--history-pages',type=int,default=CURRENT['history_pages']);p.add_argument('--local-assets');p.add_argument('--profile',action='store_true');p.add_argument('--validate',action='store_true');a=p.parse_args()
    if len(set(a.gpus))!=len(a.gpus):raise ValueError('Duplicate GPU')
    run(a)
