"""Bound stalled owned jobs; never resubmit or touch unrelated GPU processes."""
import json,time,os,signal
from .setup import ROOT
from .io import atomic_json

def main():
 path=ROOT/'watchdog.json';terminated=set()
 while True:
  try:q=json.loads((ROOT/'queue.json').read_text())
  except(FileNotFoundError,json.JSONDecodeError):time.sleep(15);continue
  if q.get('stage')in ['completed','complete_with_failures','stopped']:
   atomic_json(path,dict(status='finished',time=time.time(),terminated=sorted(terminated)));return
  for j in q.get('jobs',[]):
   if j['status']!='running'or j['id']in terminated:continue
   limit={'train':10800,'bench':1800,'baseline':1800,'evaluate':7200}[j['module']]
   if time.time()-j['started']<=limit:continue
   pid=j['pid']
   try:
    cmd=open(f'/proc/{pid}/cmdline','rb').read().replace(b'\x00',b' ').decode();expected='migration.unified_router.opt29.'+j['module']
    if expected not in cmd or os.getpgid(pid)!=pid:continue
    os.killpg(pid,signal.SIGTERM);terminated.add(j['id'])
    with(ROOT/'watchdog_events.jsonl').open('a')as f:f.write(json.dumps(dict(job=j['id'],pid=pid,reason='wall_time_limit',limit_seconds=limit,time=time.time()))+'\n')
   except(ProcessLookupError,FileNotFoundError):pass
  atomic_json(path,dict(status='monitoring',time=time.time(),terminated=sorted(terminated),retry_policy='never requeue jobs'))
  time.sleep(30)
if __name__=='__main__':main()
