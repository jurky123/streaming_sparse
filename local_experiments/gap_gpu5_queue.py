"""Run only the three explicitly listed experiments on observed idle GPUs.
No other process is stopped or modified. This is not a global GPU reservation.
"""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
from .gap_gpu_processes import inventory
ROOT=Path('artifacts/gap_gpu5_queue_20261102');ROOT.mkdir(parents=True,exist_ok=True)
JOBS=[('gap_group_fused_features_perf','/tmp/gap_group_fused_features_perf_allcontexts.log'),('gap_staged_single_perf','/tmp/gap_staged_single_perf_allcontexts.log'),('gap_tail_fused_merge_q32_check','/tmp/gap_tail_fused_merge_q32_queued.log')]
def atomic(obj):
    p=ROOT/'state.json.tmp';p.write_text(json.dumps(obj,indent=2)+'\n');p.replace(ROOT/'state.json')
def main():
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=[dict(module=m,log=l,status='pending') for m,l in JOBS];running={};idle_since={};stop=[False]
    def halt(*_):stop[0]=True
    signal.signal(signal.SIGINT,halt);signal.signal(signal.SIGTERM,halt)
    state=dict(pid=os.getpid(),status='waiting',rows=rows,policy='GPU5 released from user-authorized vLLM shutdown; 15s no compute OR graphics processes before launch; one owned process per GPU; no interference with other processes.')
    try:
        data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        uuids={int(x.split(',')[0]):x.split(',')[1].strip() for x in data.splitlines()}
        while not stop[0]:
            for gpu,(proc,row,logfile) in list(running.items()):
                code=proc.poll()
                if code is not None:
                    row.update(status='process_finished' if code==0 else 'failed',returncode=code,finished=time.time());logfile.close();del running[gpu];idle_since.pop(gpu,None)
            pending=[r for r in rows if r['status']=='pending']
            if not pending and not running:state['status']='finished';break
            try:
                busy={uuid for uuid,ps in inventory(5).items() if ps};state.pop('poll_error',None)
            except Exception as exc:
                state['poll_error']=repr(exc);idle_since.clear();atomic(state);time.sleep(5);continue
            now=time.monotonic()
            for gpu in [5]:
                if gpu in running or uuids[gpu] in busy:idle_since.pop(gpu,None);continue
                idle_since.setdefault(gpu,now)
                if pending and now-idle_since[gpu]>=15:
                    row=pending.pop(0);env=os.environ|dict(CUDA_VISIBLE_DEVICES=str(gpu),GAP_TAIL_MODE='1')
                    if row['module'].endswith('_perf'):
                        env['GAP_HOT_PAIR_ROOT']='artifacts/'+row['module']+'_allcontexts_'+str(int(time.time()))
                        row['output_root']=env['GAP_HOT_PAIR_ROOT']
                    logfile=open(row['log'],'w')
                    proc=subprocess.Popen(['bash','local_setup/run.sh','-m','local_experiments.'+row['module']],env=env,stdout=logfile,stderr=subprocess.STDOUT)
                    row.update(status='running',gpu=gpu,pid=proc.pid,started=time.time());running[gpu]=(proc,row,logfile);idle_since.pop(gpu,None)
            state['status']='running' if running else 'waiting_for_idle_gpu';state['last_poll']=time.time();atomic(state);time.sleep(5)
        if stop[0]:state['status']='stopping'
    finally:
        if stop[0]:
            for proc,row,logfile in running.values():
                if proc.poll() is None:proc.terminate()
                row['status']='termination_requested';logfile.close()
        atomic(state);lock.close()
if __name__=='__main__':main()
