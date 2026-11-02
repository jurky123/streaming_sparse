"""Resumable local experiment queue. Never benchmark alongside a peer GPU job.
This executes fixed experiments only; it does not automatically deploy candidates
or claim a theoretical speed target has been met.
"""
import argparse,fcntl,json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path('artifacts/gap_hotspots_20261031/queue')
JOBS=['gap_hot_pair','gap_route_memory_check','gap_flat_check']
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)
def devices():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    peers={}
    for line in processes.splitlines():
        uuid,pid=[v.strip() for v in line.split(',')];peers.setdefault(uuid,[]).append(int(pid))
    result=[]
    for line in rows.splitlines():
        index,uuid,memory,util=[v.strip() for v in line.split(',')]
        result.append(dict(index=index,uuid=uuid,memory_mib=int(memory),util=int(util),pids=peers.get(uuid,[])))
    return result
def free(rows):return [r for r in rows if not r['pids'] and r['memory_mib']<1024 and r['util']<5]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inspect',action='store_true');args=ap.parse_args()
    if args.inspect:print(json.dumps(dict(devices=devices(),eligible=free(devices()))));return
    ROOT.mkdir(parents=True,exist_ok=True);fd=(ROOT/'run.lock').open('a+')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fd.seek(0);fd.truncate();fd.write(str(os.getpid()));fd.flush()
    path=ROOT/'state.json';state=json.loads(path.read_text()) if path.exists() else dict(jobs={})
    stopping=[False];child=None
    def stop(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def save(status,**kw):
        state.update(status=status,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);atomic(path,state)
    try:
        for job in JOBS:
            if state['jobs'].get(job,{}).get('status') in ['complete','failed']:continue
            while not stopping[0]:
                if shutil.disk_usage('.').free<50*2**30:save('waiting_for_disk',job=job);time.sleep(30);continue
                total=sum(p.stat().st_size for p in ROOT.parent.rglob('*') if p.is_file())
                if total>2**30:save('storage_limit',job=job,bytes=total);return
                candidates=free(devices())
                if not candidates:save('waiting_for_idle_gpu',job=job);time.sleep(30);continue
                gpu=candidates[0]['index'];save('checking_idle_gpu',job=job,gpu=gpu);time.sleep(30)
                if stopping[0]:break
                if gpu not in [r['index'] for r in free(devices())]:continue
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
                env['GAP_HOT_PAIR_ROOT']=str(ROOT.parent/('hot_pair_gpu'+gpu))
                log=ROOT/(job+'.log');save('running',job=job,gpu=gpu)
                with log.open('w') as stream:
                    child=subprocess.Popen(['bash','local_setup/run.sh','-m','local_experiments.'+job],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    contaminated=False
                    while child.poll() is None and not stopping[0]:
                        time.sleep(5)
                        row=next(r for r in devices() if r['index']==gpu)
                        others=[pid for pid in row['pids'] if pid!=child.pid]
                        if others:
                            contaminated=True;save('interrupted_by_peer',job=job,gpu=gpu,peer_pids=others);break
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    code=child.returncode;child=None
                if stopping[0]:break
                if contaminated:continue
                state['jobs'][job]=dict(status='complete' if code==0 else 'failed',exit_code=code,gpu=gpu,log=str(log))
                save('job_finished',job=job,gpu=gpu);break
            if stopping[0]:save('interrupted',job=job);return
        save('experiments_finished',note='Requires result analysis and real quality acceptance for any numerically changed candidate; theoretical target not certified.')
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        fd.close()
if __name__=='__main__':main()
