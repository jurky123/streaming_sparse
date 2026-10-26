"""Detached collection -> local training -> actual29-layer quality evaluation."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2];PY=ROOT/'streaming_sparse/.venv/bin/python'


def read(path):
    try:return json.loads(Path(path).read_text())
    except(FileNotFoundError,ValueError):return{}


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve()
    lock=(root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logfiles=[];child=None;current={};active=None
    def status(**kw):
        current.update(kw,controller_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat())
        tmp=root/'status.tmp';tmp.write_text(json.dumps(current,indent=2)+'\n');tmp.replace(root/'status.json');print(json.dumps(current),flush=True)
    def stop(signum,frame):(root/'STOP').touch();raise_stop()
    def raise_stop():raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(module,gpus,logdir,extra=()):
        nonlocal child,active
        if(root/'STOP').exists():raise KeyboardInterrupt()
        logdir=Path(logdir);logdir.mkdir(parents=True,exist_ok=True);active=logdir
        st=read(logdir/'launcher_status.json')
        if st.get('status')=='completed':return
        if st.get('status')=='running'and Path(f"/proc/{st['launcher_pid']}").exists():child=None
        else:
            f=(logdir/'controller_stdout.log').open('a');logfiles.append(f)
            env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
            child=subprocess.Popen([str(PY),'-u',str(ROOT/'migration/runtime/launch.py'),'--gpus',','.join(map(str,gpus)),
                '--module',module,'--logdir',str(logdir),'--',*map(str,extra)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        while True:
            if(root/'STOP').exists():raise KeyboardInterrupt()
            st=read(logdir/'launcher_status.json')
            if st.get('status')=='completed':return
            if st.get('status')in ['failed','stopped']:raise RuntimeError(f'{module} failed; {logdir}')
            if child is not None and child.poll()is not None and child.returncode!=0:raise RuntimeError(f'Launcher failed: {module}')
            time.sleep(3)
    try:
        for phase,module,gpus in [('audit','migration.local_router.audit',[0]),('benchmark','migration.local_router.train',[0]),
                ('collection','migration.local_router.collect',list(range(8))),('training','migration.local_router.train',list(range(8)))]:
            status(status='running',stage=phase)
            args=['--directory',root]+(['--benchmark']if phase=='benchmark'else[])
            launch(module,gpus,root/f'{phase}_workers',args)
        status(status='running',stage='assembling')
        launch('migration.local_router.finish',[0],root/'assembly_workers',['--directory',root])
        bridge=root/'network_check';status(status='running',stage='network_quality')
        launch('migration.paired_router.quality',list(range(8)),root/'quality_workers',['--directory',bridge,'--stage','dense0'])
        launch('migration.paired_router.evaluation',[0],root/'quality_merge_workers',['--directory',bridge,'--stage','dense0','--merge'])
        report=bridge/'dense0/quality/REPORT.md'
        text=report.read_text();text=text.replace('骨干和LoRA全冻结；K/V同预算、同数据顺序。',
            '本实验仅用逐层Attention输出NMSE训练Router；冻结Dense骨干采集输入，各层独立300–800更新，K/V配对停止。这里的x0/视频是训练后整网评估，不是训练损失。逐层最佳权重由各层局部Val分别选择。骨干和LoRA全冻结；K/V同预算、同数据顺序。')
        text=text.replace('精确停止判据与两组平台情况见各自stop_decision.json。','逐层停止判据与更新数见主实验的training/layerXX/validation.json和assembled.json。')
        report.write_text(text);status(status='completed',stage='all_finished',report=str(report))
    except KeyboardInterrupt:status(status='paused')
    except BaseException as error:status(status='failed',error=repr(error));raise
    finally:
        if current.get('status')!='completed'and active:
            st=read(active/'launcher_status.json');pid=st.get('launcher_pid')
            if st.get('status')=='running'and pid:
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for f in logfiles:f.close()


if __name__=='__main__':main()
