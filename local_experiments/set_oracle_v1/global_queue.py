"""Pre-registered small certificate panel; bounded phases, no outcome-driven sample changes."""
import os,time,json,subprocess,sys
from .common import *
if __name__=='__main__':
 manifest=json.loads((ROOT/'manifest.json').read_text());registered=manifest['global_preregistered'];active={};initial_done=set();extended=set();audited=set();failures=[];started=time.monotonic();cpubudget=12*3600
 def launch(eid,phase):
  module={'initial':'global_bound','extend':'parallel_bound','audit_initial':'audit','audit_extended':'audit'}[phase];args=[sys.executable,'-m','local_experiments.set_oracle_v1.'+module,'--event',eid]
  if phase=='initial':args+=['--seconds','90']
  if phase=='extend':args+=['--workers','16','--seconds','150']
  if phase=='audit_extended':args+=['--root',str(ROOT/'global_bounds_parallel')]
  log=(ROOT/'logs'/f'{phase}_{eid}.log').open('w');proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);active[proc.pid]=(proc,eid,phase,log);print('LAUNCH',phase,eid,proc.pid,flush=True)
 def used_cpu():
  total=0.
  for eid in registered:
   f=ROOT/'global_bounds_parallel'/eid/'branch_result.json';f=f if f.exists() and json.loads(f.read_text()).get('parallel_workers') else ROOT/'global_bounds'/eid/'branch_result.json'
   if f.exists():total+=json.loads(f.read_text()).get('cpu_seconds',0.)
  return total
 while len(audited)+len({x['event_id'] for x in failures})<len(registered):
  for pid,(proc,eid,phase,log) in list(active.items()):
   if proc.poll() is None:continue
   log.close();del active[pid]
   if proc.returncode:failures.append(dict(event_id=eid,phase=phase,returncode=proc.returncode));continue
   if phase=='initial':initial_done.add(eid)
   elif phase=='extend':extended.add(eid)
   else:audited.add(eid)
  busy={v[1] for v in active.values()};failed={x['event_id'] for x in failures}
  for eid in registered:
   if eid in busy or eid in audited or eid in failed:continue
   if eid not in initial_done:
    if (ROOT/'searches'/'v1'/eid/'result.json').exists() and sum(v[2]=='initial' for v in active.values())<12:launch(eid,'initial')
   else:
    result=json.loads((ROOT/'global_bounds'/eid/'branch_result.json').read_text())
    if result['status']=='GLOBAL_TOL':launch(eid,'audit_initial')
    elif eid in extended:launch(eid,'audit_extended')
    elif sum(v[2]=='extend' for v in active.values())<3:
     # Reserve maximum16*180 CPU-seconds per unfinished extension, leaving audit headroom.
     reserved=sum(v[2]=='extend' for v in active.values())*16*180
     if used_cpu()+reserved+16*180<cpubudget-1800:launch(eid,'extend')
     else:launch(eid,'audit_initial')
  dump(ROOT/'global_queue_status.json',dict(registered=registered,initial_finished=len(initial_done),extensions_finished=len(extended),audited=len(audited),active=[dict(pid=k,event_id=v[1],phase=v[2]) for k,v in active.items()],failures=failures,accounted_solver_cpu_seconds=used_cpu(),cpu_cap_hours=12,wall_seconds=time.monotonic()-started))
  if time.monotonic()-started>14400:raise RuntimeError('global queue exceeded wall guard')
  time.sleep(2)
 dump(ROOT/'global_queue_status.json',dict(status='complete' if not failures else 'failed',registered=registered,audited=len(audited),failures=failures,accounted_solver_cpu_seconds=used_cpu(),cpu_cap_hours=12,wall_seconds=time.monotonic()-started))
