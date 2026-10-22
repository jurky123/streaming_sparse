"""Observe completion; optionally resume externally preempted jobs on idle GPUs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from .monitor import processes,read_json,write_json

HOME=Path(__file__).resolve().parent
ROOT=HOME.parents[1]
LOGS=HOME/'logs'


def alive(path,module):
    try:
        proc=Path('/proc')/path.read_text().strip()
        return module.encode() in (proc/'cmdline').read_bytes().split(b'\0')
    except OSError:return False


def event(kind,**fields):
    value=dict(time_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),pid=os.getpid(),event=kind,**fields)
    with (LOGS/'completion_watch.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    write_json(LOGS/'completion_watch.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def progress_signature():
    values=[]
    for directory in sorted((HOME/'runs/full').glob('*/seed*')):
        latest=directory/'latest.pt'
        values.append((str(directory),latest.stat().st_mtime_ns if latest.exists() else 0))
        for split in ('train','val','test'):
            path=directory/f'eval_{split}'
            values.append(((read_json(path/'progress.json') or {}).get('snapshots_done',0),
                           (path/'summary.json').exists()))
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def ensure_monitor():
    if alive(HOME/'monitor.pid','streaming_sparse.v02.monitor'):return
    with (LOGS/'monitor.log').open('a') as f:
        subprocess.Popen([sys.executable,'-u','-m','streaming_sparse.v02.monitor'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)


def launch(gpus):
    name='self-forcing-sparse-v02'
    args=[sys.executable,'-u','-m','streaming_sparse.v02.supervisor','--gpu',str(gpus[-1]),
          '--gpus',*map(str,gpus),'--max-restarts','2']
    command=shlex.join(args)+' >> '+shlex.quote(str(LOGS/'supervisor.log'))+' 2>&1'
    exists=subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    if exists:
        dead=subprocess.check_output(['tmux','display-message','-p','-t',name+':0.0','#{pane_dead}'],text=True).strip()
        if dead!='1':raise RuntimeError('Main tmux pane is still alive; refusing duplicate launch')
    subprocess.run(['tmux','respawn-pane','-t',name+':0.0','-c',str(ROOT),command] if exists else
                   ['tmux','new-session','-d','-s',name,'-c',str(ROOT),command],check=True)
    subprocess.run(['tmux','set-option','-t',name,'remain-on-exit','on'],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs='+',type=int,default=[1,2,3,4,5,6,7])
    parser.add_argument('--auto-resume',action='store_true')
    args=parser.parse_args()
    with (LOGS/'completion_watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (LOGS/'completion_watch.pid').write_text(str(os.getpid())+'\n')
        quiet_since=None;last_progress=progress_signature();without_progress=0
        event('watching',auto_resume=args.auto_resume,gpus=args.gpus)
        while True:
            if (HOME/'supervisor.stop').exists():event('explicit_stop');return
            ready=read_json(HOME.parent/'artifacts/v02_REVIEW_COMPLETE.json')
            if ready and ready.get('complete'):
                if ready.get('audited_training_runs')!=15 or ready.get('audited_evaluations')!=45:
                    raise RuntimeError('Review completion counts mismatch')
                if not all(Path(ready[k]['path']).exists() and ready[k]['crc_verified'] for k in ('review','raw_results')):
                    raise RuntimeError('Review completion marker without verified archives')
                target=HOME/'reports/REVIEW_READY.md'
                target.write_text('# v0.2 评审包已完成\n\n15 组 Full 训练、45 次完整评估与验收已完成。\n\n'+
                    f"- [主评审包]({HOME.parent/'artifacts/v02_review_latest.zip'})\n"+
                    f"- [逐 Tile 原始结果]({HOME.parent/'artifacts/v02_raw_results_latest.zip'})\n"+
                    f"- [完成记录与 SHA-256]({HOME.parent/'artifacts/v02_REVIEW_COMPLETE.json'})\n")
                event('review_ready',packages=ready)
                return
            ensure_monitor()
            state=read_json(HOME/'status.json') or {}
            if alive(HOME/'workflow.pid','streaming_sparse.v02.workflow') or alive(HOME/'supervisor.pid','streaming_sparse.v02.supervisor'):
                quiet_since=None;time.sleep(30);continue
            signature=progress_signature()
            if signature!=last_progress:without_progress=0;last_progress=signature
            detail=state.get('detail','')
            recoverable=(state.get('state') in ('running','waiting_resources') or
                         (state.get('state')=='failed' and detail.startswith('Supervisor received external signal')
                          and ('signal 1;' in detail or 'signal 15;' in detail)))
            if not args.auto_resume or not recoverable:
                event('needs_inspection',status=state);return
            owned=[]
            for p in processes():
                try:
                    if (Path('/proc')/str(p['pid'])/'cwd').resolve()==ROOT.resolve():owned.append(p)
                except OSError:pass
            if owned:quiet_since=None;time.sleep(30);continue
            try:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                memory={int(a):int(b) for a,b in (line.split(',') for line in rows.splitlines())}
            except (OSError,subprocess.CalledProcessError,ValueError) as error:
                event('resource_query_failed',error=str(error));time.sleep(30);continue
            idle=[g for g in args.gpus if memory.get(g,999999)<512]
            if not idle:
                quiet_since=None
                if state.get('state')!='waiting_resources':
                    write_json(HOME/'status.json',dict(phase=state.get('phase','P2/P3 Full multi-GPU'),
                        detail=f'Waiting for idle candidate GPUs {args.gpus}; checkpoints preserved; automatic resume enabled',
                        state='waiting_resources',updated_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())))
                    event('waiting_resources',gpu_memory_mib=memory,previous_status=state)
                time.sleep(30);continue
            quiet_since=quiet_since or time.monotonic()
            if time.monotonic()-quiet_since<30:time.sleep(30);continue
            if without_progress>=3:event('no_progress_restart_limit',status=state);return
            try:
                launch(args.gpus);without_progress+=1
                event('external_preemption_resumed',idle_gpus=idle,attempts_without_progress=without_progress)
            except (OSError,subprocess.CalledProcessError,RuntimeError) as error:
                without_progress+=1;event('launch_failed',error=str(error))
            quiet_since=None;time.sleep(30)


if __name__=='__main__':main()
