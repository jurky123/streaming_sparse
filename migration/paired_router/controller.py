"""Durable staged execution; failure stops the queue; never resumes the old run."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];PY=ROOT/'streaming_sparse/.venv/bin/python'


def read(path):
    try:return json.loads(Path(path).read_text())
    except(FileNotFoundError,ValueError):return{}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve();manifest=read(root/'experiment.json')
    lock=(root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children=[];open_logs=[];current={}
    def status(**kw):
        current.update(kw,controller_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat());write(root/'status.json',current)
        print(json.dumps(current,ensure_ascii=False),flush=True)
    def stopped():return(root/'STOP').exists()
    def stop(signum,frame):
        (root/'STOP').touch();raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(module,gpus,logdir,args):
        logdir=Path(logdir);existing=read(logdir/'launcher_status.json')
        if existing.get('status')=='running'and Path(f"/proc/{existing['launcher_pid']}").exists():return logdir
        if existing.get('status')=='completed':return logdir
        logdir.mkdir(parents=True,exist_ok=True);stream=(logdir/'controller_stdout.log').open('a');open_logs.append(stream)
        env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        command=[str(PY),'-u',str(ROOT/'migration/runtime/launch.py'),'--gpus',','.join(map(str,gpus)),'--module',module,'--logdir',str(logdir),'--',*map(str,args)]
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT);children.append(child)
        return logdir
    def wait_jobs(jobs):
        while True:
            if stopped():raise KeyboardInterrupt()
            values=[read(job/'launcher_status.json')for job in jobs]
            if any(v.get('status')in ['failed','stopped']for v in values):raise RuntimeError(f'Worker phase failed; inspect {jobs}')
            if all(v.get('status')=='completed'for v in values):return
            for v in values:
                if v.get('status')=='running'and not Path(f"/proc/{v['launcher_pid']}").exists():raise RuntimeError('Launcher exited without completing')
            time.sleep(5)
    def cpu(module,args):
        if stopped():raise KeyboardInterrupt()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        subprocess.run([str(PY),'-u','-m','migration.runtime.bootstrap',module,*map(str,args)],cwd=ROOT,env=env,check=True)
    try:
        for stage in ['three','dense0']:
            directory=Path(manifest['stages'][stage]['directory'])
            if stage=='dense0'and not(directory/'initialization_manifest.json').exists():
                status(status='running',stage='calibrate_dense0')
                wait_jobs([launch('migration.paired_router.calibrate',list(range(8)),directory/'calibration_workers',['--directory',root])])
                cpu('migration.paired_router.assemble',['--directory',root])
            status(status='running',stage='audit_'+stage)
            jobs=[]
            for group,gpus in manifest['physical_gpus'].items():
                report=read(directory/f'audits/{group}.json')
                if report.get('passed'):continue
                jobs.append(launch('migration.paired_router.audit',gpus,directory/f'audit_workers_{group}',['--directory',root,'--stage',stage,'--group',group]))
            if jobs:wait_jobs(jobs)
            assert all(read(directory/f'audits/{g}.json').get('passed')for g in manifest['groups'])
            status(status='running',stage='train_'+stage)
            jobs=[]
            for group,gpus in manifest['physical_gpus'].items():
                run=directory/f'runs/seed0/{group}'
                if(run/'completed.json').exists():continue
                resume=run/'recovery.pt'if(run/'recovery.pt').exists()else directory/f'initial_{group[-1]}.pt'
                args=['--directory',root,'--stage',stage,'--resume',resume,'--group',group,
                    '--audit',ROOT/'migration/restoration/20261022_133350/router_speed_audit.json','--parallel-audit',directory/f'audits/{group}.json',
                    '--batch-size','8','--batch-amendment',ROOT/'migration/restoration/20261022_133350/batch8_amendment.json','--status',directory/f'{group}_live.json']
                jobs.append(launch('migration.paired_router.entry',gpus,directory/f'train_workers_{group}',args))
            if jobs:wait_jobs(jobs)
            assert all((directory/f'runs/seed0/{g}/last.pt').exists()for g in manifest['groups'])
            status(status='running',stage='quality_'+stage)
            cpu('migration.paired_router.evaluation',['--directory',root,'--stage',stage])
            wait_jobs([launch('migration.paired_router.quality',list(range(8)),directory/'quality_workers',['--directory',root,'--stage',stage])])
            cpu('migration.paired_router.evaluation',['--directory',root,'--stage',stage,'--merge'])
        status(status='completed',stage='all_finished')
    except KeyboardInterrupt:status(status='paused',stage=current.get('stage'))
    except BaseException as error:status(status='failed',error=repr(error));raise
    finally:
        if current.get('status')!='completed':
            for stage in ['three','dense0']:
                directory=Path(manifest['stages'][stage]['directory'])
                for path in directory.glob('*/launcher_status.json'):
                    value=read(path);pid=value.get('launcher_pid');cmd=Path(f'/proc/{pid}/cmdline')
                    if value.get('status')=='running'and cmd.exists()and b'migration/runtime/launch.py'in cmd.read_bytes():
                        try:os.kill(pid,signal.SIGTERM)
                        except ProcessLookupError:pass
        for stream in open_logs:stream.close()


if __name__=='__main__':main()
