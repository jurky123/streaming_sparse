import concurrent.futures,time,json,traceback,multiprocessing
from .common import *
from .search import run
if __name__=='__main__':
 manifest=json.loads((ROOT/'manifest.json').read_text());expected={e['event_id'] for e in manifest['events']};seen=set();pending={};failures=[];begin=time.monotonic()
 with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn')) as pool:
  while len(seen)<len(expected) or pending:
   for path in (ROOT/'contributions').glob('*/READY.json'):
    eid=path.parent.name
    if eid not in seen and len(pending)<24:seen.add(eid);pending[pool.submit(run,eid)]=eid
   for f in list(pending):
    if f.done():
     eid=pending.pop(f)
     try:f.result()
     except Exception as exc:failures.append(dict(event_id=eid,error=repr(exc),traceback=''.join(traceback.format_exception(exc))));print('SEARCH_FAILED',eid,repr(exc),flush=True)
   done=sum((ROOT/'searches'/'v1'/eid/'result.json').exists() for eid in expected);dump(ROOT/'search_queue_status.json',dict(expected=len(expected),seen=len(seen),completed=done,active=len(pending),failures=failures,wall_seconds=time.monotonic()-begin))
   if time.monotonic()-begin>14400:raise RuntimeError('queue guard: capture incomplete after4h')
   time.sleep(2)
 dump(ROOT/'search_queue_status.json',dict(expected=len(expected),seen=len(seen),completed=len(expected)-len(failures),failures=failures,status='complete' if not failures else 'failed'))
