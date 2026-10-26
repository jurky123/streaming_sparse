"""Durable eight-GPU queue. One attempt per job; independent failures do not retry."""
import json,os,time,subprocess,sys,fcntl,signal
from pathlib import Path
from .setup import ROOT,PILOTS,VARIANTS
from .io import atomic_json
PYTHON=Path(__file__).resolve().parents[3]/'streaming_sparse/.venv/bin/python'
def main():
 ROOT.mkdir(parents=True,exist_ok=True);lock=open(ROOT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((ROOT/'preflight.json').read_text())['passed']
 path=ROOT/'queue.json';state=json.loads(path.read_text())if path.exists()else dict(started=time.time(),stage='pilot',jobs=[],promoted=[],video_queued=False)
 jobs=state['jobs'];running={}
 for j in jobs:
  if j['status']=='running':j.update(status='orphaned',error='Supervisor restarted; no automatic duplicate/retry')
 def add(id,module,args,dependency=None):
  if any(j['id']==id for j in jobs):return
  jobs.append(dict(id=id,module=module,args=args,dependency=dependency,status='pending'))
 for name in VARIANTS:
  for layer in PILOTS:
   id=f'pilot_{name}_{layer:02d}';add(id,'train',['--layer',str(layer),'--variant',name,'--updates','2000','--minimum','1000'])
   add('bench_'+id,'bench',['--layer',str(layer),'--variant',name],id)
 for layer in PILOTS:add(f'baseline_{layer:02d}','baseline',['--layer',str(layer)])
 def record():state.update(pid=os.getpid(),updated=time.time(),workers={str(g):j['id']for g,(p,f,j)in running.items()});atomic_json(path,state)
 try:
  while True:
   stopped=(ROOT/'STOP').exists()
   for gpu,(p,f,j)in list(running.items()):
    rc=p.poll()
    if rc is not None:j.update(status='done'if rc==0 else'failed',exit_code=rc,finished=time.time());f.close();del running[gpu];record()
   byid={j['id']:j for j in jobs}
   for j in jobs:
    if j['status']=='pending'and j['dependency']and byid[j['dependency']]['status']in ['failed','blocked','orphaned']:j.update(status='blocked',error='Prerequisite did not complete')
   ready=(ROOT/'POSTPROCESS_READY').exists()
   terminal=lambda js:all(j['status']in ['done','failed','blocked','orphaned']for j in js)
   if ready and state['stage']=='pilot'and terminal([j for j in jobs if j['id'].startswith(('pilot_','bench_pilot_','baseline_'))]):
    from .promote import choose
    selected=choose();state.update(stage='final',promoted=selected)
    for name in selected:
     for layer in range(1,30):
      id=f'final_{name}_{layer:02d}';add(id,'train',['--layer',str(layer),'--variant',name,'--updates','6000','--minimum','2000']);add('bench_'+id,'bench',['--layer',str(layer),'--variant',name,'--stage','final'],id)
    record()
   if ready and state['stage']=='final'and terminal([j for j in jobs if j['id'].startswith(('final_','bench_final_'))]):
    viable=[name for name in state['promoted']if all(byid.get(f'final_{name}_{l:02d}',{}).get('status')=='done'for l in range(1,30))]
    state.update(stage='video',viable=viable)
    for height,width,cases in [(480,832,4),(720,1280,4)]:
     for case in range(cases):add(f'video_{height}_{case}','evaluate',['--height',str(height),'--width',str(width),'--case',str(case)])
    record()
   if ready and state['stage']=='video'and terminal([j for j in jobs if j['id'].startswith('video_')]):
    state['stage']='complete_with_failures'if any(j['status']in['failed','blocked','orphaned']for j in jobs)else'completed';record();break
   if stopped and not running:state['stage']='stopped';record();break
   byid={j["id"]:j for j in jobs}
   if not stopped:
    for gpu in range(8):
     reserved=json.loads((ROOT/'reserved_gpus.json').read_text())if(ROOT/'reserved_gpus.json').exists()else[]
     if gpu in running or gpu in reserved:continue
     candidates=[j for j in jobs if j['status']=='pending'and(j['dependency']is None or byid[j['dependency']]['status']=='done')and(ready or j['module']=='train')]
     if not candidates:break
     j=candidates[0];logs=ROOT/'logs';logs.mkdir(exist_ok=True);log=logs/(j['id']+'.log');f=log.open('a',buffering=1)
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='2',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='1')
     cmd=[str(PYTHON),'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt29.'+j['module']]+j['args'];p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     j.update(status='running',gpu=gpu,pid=p.pid,started=time.time(),log=str(log),command=cmd);running[gpu]=(p,f,j);record()
   if ready:
    try:
     from .report import generate
     generate(state)
    except Exception as e:state['report_error']=repr(e)
   record();time.sleep(5)
 finally:
  record()
  if(ROOT/'POSTPROCESS_READY').exists():
   from .report import generate
   generate(state)
if __name__=='__main__':main()
