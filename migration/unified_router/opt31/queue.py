import os,json,subprocess,time,fcntl,signal
from pathlib import Path
from .setup import ROOT,PARENT
from ..opt30.io import atomic_json
PYTHON=ROOT.parents[2]/'.venv/bin/python'
def main():
 ROOT.mkdir(parents=True,exist_ok=True);lock=open(ROOT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 path=ROOT/'queue.json';state=json.loads(path.read_text())if path.exists()else dict(stage='preflight',jobs=[]);jobs=state['jobs'];running={}
 class Adopted:
  def __init__(self,j):self.pid=j['pid'];self.job=j;self.returncode=None
  def poll(self):
   p=Path(f'/proc/{self.pid}/stat')
   if p.exists()and p.read_text().split(') ',1)[1].split()[0]!='Z':return None
   args=self.job['args'];get=lambda k:args[args.index(k)+1]
   if self.job['module']=='preflight':result=ROOT/f'preflight_{get("--height")}.json';key='passed'
   else:result=ROOT/f'e2e/{get("--seconds")}s/{get("--height")}x{get("--width")}/case{get("--case")}/result.json';key='completed'
   self.returncode=0 if result.exists()and json.loads(result.read_text()).get(key)else 1
   return self.returncode
 for j in jobs:
  if j['status']=='running':
   cmd=Path(f'/proc/{j["pid"]}/cmdline')
   if cmd.exists():assert b'migration.unified_router.opt31.' in cmd.read_bytes()
   f=open(j['log'],'a',buffering=1);running[j['gpu']]=(Adopted(j),f,j)
 if not jobs:
  manifest=json.loads((ROOT/'manifest.json').read_text());configs=manifest['configurations']
  for h,w in sorted({(c['height'],c['width'])for c in configs}):jobs.append(dict(id=f'preflight_{h}x{w}',module='preflight',args=['--height',str(h),'--width',str(w)],status='pending'))
  for c in configs:
   h,w,duration=c['height'],c['width'],c['seconds']
   for case in range(manifest['cases']):jobs.append(dict(id=f'eval_{duration}s_{h}x{w}_{case}',module='evaluate',args=['--height',str(h),'--width',str(w),'--seconds',str(duration),'--case',str(case)],status='pending'))
 def record():state.update(pid=os.getpid(),updated=time.time());atomic_json(path,state)
 try:
  while True:
   parent=json.loads((PARENT/'queue.json').read_text());occupied={j['gpu']for j in parent['jobs']if j['status']=='running'}
   for gpu,(p,f,j)in list(running.items()):
    if p.poll()is not None:j.update(status='done'if p.returncode==0 else'failed',exit_code=p.returncode,finished=time.time());f.close();del running[gpu]
    elif time.time()-j['started']>7200:os.killpg(p.pid,signal.SIGTERM)
   pre=[j for j in jobs if j['module']=='preflight'];ready=all(j['status']=='done'for j in pre)
   if any(j['status']=='failed'for j in pre):
    for j in jobs:
     if j['status']=='pending':j.update(status='blocked',reason='Long-video correctness preflight failed')
   state['stage']='evaluation'if ready else'preflight'
   for gpu in range(8):
    if gpu in occupied or gpu in running:continue
    pending=[j for j in jobs if j['status']=='pending'and(ready or j['module']=='preflight')]
    if not pending:break
    j=pending[0];log=ROOT/'logs'/f'{j["id"]}.log';log.parent.mkdir(exist_ok=True);f=log.open('a',buffering=1);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='2',OMP_NUM_THREADS='4')
    cmd=[str(PYTHON),'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt31.'+j['module']]+j['args'];p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    j.update(status='running',gpu=gpu,pid=p.pid,started=time.time(),log=str(log));running[gpu]=(p,f,j);record()
   if all(j['status']in ['done','failed','blocked','superseded']for j in jobs):state['stage']='completed'if all(j['status']in ['done','superseded']for j in jobs)else'complete_with_failures';record();break
   record()
   from .report import main as report
   report();time.sleep(5)
 finally:
  record()
  if not running:atomic_json(PARENT/'reserved_gpus.json',[])
  from .report import main as report
  report()
if __name__=='__main__':main()
