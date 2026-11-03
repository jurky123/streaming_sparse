"""CPU-only deferred submission to our existing session; never touches peers."""
import json,os,time
from pathlib import Path
from .gap_gpu_processes import inventory
SESSION=Path('artifacts/gap_stream_session_20261102')
ROOT=Path('artifacts/review_v1_20261102')
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def main():
    ROOT.mkdir(exist_ok=True)
    # Do not race another queue monitor.
    import fcntl
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[('probe_initial','gap_review_v1_probe',['--case','bird_flight']),
          ('summary_pilot','gap_review_v1_train',[]),
          ('probe_trained','gap_review_v1_probe',['--case','bird_flight','--tag','summary_pilot','--summary-checkpoint','artifacts/review_v1_20261102/summary_train/pilot/latest.pt']),
          ('calibration_data','gap_review_v1_probe',['--case','city_walk','--tag','summary_pilot','--summary-checkpoint','artifacts/review_v1_20261102/summary_train/pilot/latest.pt']),
          ('calibration_fit','gap_review_v1_calibrate',[]),
          ('probe_route_trained','gap_review_v1_probe',['--case','bird_flight','--tag','route_pilot','--summary-checkpoint','artifacts/review_v1_20261102/route_pilot/model.pt'])]
    for name,module,args in jobs:
        if (ROOT/(name+'.queue_done.json')).exists():continue
        while True:
            state=json.loads((SESSION/'status.json').read_text());owner=state['pid']
            try:os.kill(owner,0)
            except ProcessLookupError:
                atomic(ROOT/'queue_status.json',dict(status='blocked_session_not_alive',session_pid=owner));return
            peers=[r for ps in inventory(str(state['gpu'])).values() for r in ps if r['pid']!=owner]
            if peers or state['status']!='reserved_waiting':
                atomic(ROOT/'queue_status.json',dict(status='waiting_for_exclusive_session',next=name,peers=peers,updated=time.time(),pid=os.getpid()));time.sleep(30);continue
            uid='review_v1_'+name+'_'+str(time.time_ns())
            tasks=json.loads((SESSION/'tasks.json').read_text());tasks.append(dict(id=uid,module=module,args=args));atomic(SESSION/'tasks.json',tasks)
            atomic(ROOT/'queue_status.json',dict(status='submitted',task=uid,updated=time.time(),pid=os.getpid()))
            result=SESSION/(uid+'.result.json')
            while not result.exists():time.sleep(10)
            row=json.loads(result.read_text())
            if row['status']=='failed':
                if 'shared before task' in row.get('error',''):continue
                atomic(ROOT/'queue_status.json',dict(status='failed',task=uid,result=row));return
            atomic(ROOT/(name+'.queue_done.json'),row);break
    atomic(ROOT/'queue_status.json',dict(status='pilot_jobs_complete_pending_review',updated=time.time()))
if __name__=='__main__':main()
