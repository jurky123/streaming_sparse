import json,time,os,fcntl,subprocess,hashlib,signal
from .setup import ROOT,PARENT,PRIORITY,ANCHOR,PILOTS,VARIANTS
from ..opt30.io import atomic_json
PYTHON=ROOT.parents[2]/'.venv/bin/python'
def main():
 ROOT.mkdir(parents=True,exist_ok=True);lock=open(ROOT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=ROOT/'queue.json'
 state=json.loads(path.read_text())if path.exists()else dict(stage='waiting_priority_and_opt30',jobs=[]);jobs=state['jobs'];running={}
 if any(j['status']=='running'for j in jobs):raise RuntimeError('Reconcile existing jobs before restarting')
 if not jobs:
  for layer in PILOTS:jobs.append(dict(id=f'prepare_{layer:02d}',args=['--layer',str(layer),'--prepare'],dependency=None,status='pending'))
  for name in VARIANTS:
   for layer in PILOTS:jobs.append(dict(id=f'{name}_{layer:02d}',args=['--layer',str(layer),'--variant',name,'--steps','1200'],dependency=f'prepare_{layer:02d}',status='pending'))
 def record():state.update(pid=os.getpid(),updated=time.time());atomic_json(path,state)
 record()
 while state['stage']=='waiting_priority_and_opt30':
  pp=json.loads((PRIORITY/'queue.json').read_text())if(PRIORITY/'queue.json').exists()else{};parent=json.loads((PARENT/'queue.json').read_text())
  if pp.get('stage')in ['completed','complete_with_failures']and parent.get('stage')in ['completed','complete_with_failures']:
   required=[next((j for j in parent['jobs']if j['id']==f'final_{ANCHOR}_{l:02d}'),{})for l in range(1,30)]
   if not all(j.get('status')=='done'for j in required):
    state['stage']='blocked_missing_complete_foundation'
    for j in jobs:j.update(status='blocked',reason='Anchor must complete all29 layers before paired action study')
    record();return
   checks={}
   for layer in range(1,30):
    p=PARENT/f'training/layer{layer:02d}/{ANCHOR}/best.pt';checks[str(layer)]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
   atomic_json(ROOT/'foundation.json',dict(anchor=ANCHOR,checkpoints=checks,base_frozen=True));state['stage']='pilot';record();break
  record();time.sleep(30)
 while True:
  for gpu,(p,f,j)in list(running.items()):
   if p.poll()is not None:j.update(status='done'if p.returncode==0 else'failed',exit_code=p.returncode,finished=time.time());f.close();del running[gpu]
   elif time.time()-j['started']>10800:os.killpg(p.pid,signal.SIGTERM)
  byid={j['id']:j for j in jobs}
  for j in jobs:
   if j['status']=='pending'and j['dependency']and byid[j['dependency']]['status']in ['failed','blocked']:j.update(status='blocked',reason='Calibration/native correctness prerequisite failed')
  for gpu in range(8):
   if gpu in running:continue
   ready=[j for j in jobs if j['status']=='pending'and(j['dependency']is None or byid[j['dependency']]['status']=='done')]
   if not ready:break
   j=ready[0];log=ROOT/'logs'/f'{j["id"]}.log';log.parent.mkdir(exist_ok=True);f=log.open('a',buffering=1);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='2',OMP_NUM_THREADS='4')
   p=subprocess.Popen([str(PYTHON),'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt32.worker']+j['args'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);j.update(status='running',pid=p.pid,gpu=gpu,started=time.time());running[gpu]=(p,f,j);record()
  if all(j['status']in ['done','failed','blocked']for j in jobs):state['stage']='pilot_completed'if all(j['status']=='done'for j in jobs)else'pilot_complete_with_failures';record();break
  record()
  from .report import main as report
  report();time.sleep(5)
 from .report import main as report
 report()
if __name__=='__main__':main()
