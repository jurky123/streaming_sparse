"""Two host GPUs, at most one experiment process per GPU; fail closed on Phase1."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import HERE,DEFAULT_DATA,SnapshotData,versions,source_identity,atomic_json,protect_output
from .model import Router


def prepare():
    python=str(Path(sys.executable).resolve());cwd=str(HERE.parents[1])
    # Preserve venv symlink entry point, rather than resolving out of its environment.
    python=str(HERE.parents[0]/'.venv/bin/python')
    jobs=[]
    def add(name,module,args,gpu,deps):
        jobs.append(dict(id=name,argv=[python,'-u','-m',module,*map(str,args)],gpu=gpu,depends_on=deps))
    for gpu in (0,1):
        gate=HERE/f'reports/correctness_gpu{gpu}.json'
        add(f'correctness_gpu{gpu}','streaming_sparse.v05.sanity',['--output',gate],gpu,[])
        add(f'phase1_seed{gpu}','streaming_sparse.v05.train',
            ['--phase','phase1','--seed',gpu,'--output',HERE/f'phase1/seed{gpu}','--gate',gate],gpu,[f'correctness_gpu{gpu}'])
    add('phase1_gate','streaming_sparse.v05.report',['--phase1-gate'],None,['phase1_seed0','phase1_seed1'])
    last={0:'phase1_gate',1:'phase1_gate'}
    for seed,gpu in ((0,0),(1,1),(2,0)):
        name=f'phase2_seed{seed}'
        add(name,'streaming_sparse.v05.train',['--phase','phase2','--seed',seed,'--output',HERE/f'phase2/seed{seed}',
             '--gate',HERE/f'reports/correctness_gpu{gpu}.json'],gpu,[last[gpu]])
        last[gpu]=name
        for split in ('val','test','train'):
            job=f'eval_seed{seed}_{split}'
            add(job,'streaming_sparse.v05.evaluate',['--checkpoint',HERE/f'phase2/seed{seed}/best.pt','--split',split,
                '--output',HERE/f'phase2/seed{seed}/evaluation_{split}'],gpu,[last[gpu]])
            last[gpu]=job
    add('final_report','streaming_sparse.v05.report',[],None,[last[0],last[1]])
    train=SnapshotData(DEFAULT_DATA,'train');val=SnapshotData(DEFAULT_DATA,'val')
    plan=dict(source_sha256=source_identity(),cwd=cwd,jobs=jobs,capacity=Router().capacity(),
              phase1_train_snapshots=len(versions(train,'phase1','train')),
              phase1_val_snapshots=len(versions(val,'phase1','monitor')),
              phase2_train_snapshots=len(train.groups),phase2_monitor_snapshots=len(versions(val,'phase2','monitor')),
              data_fingerprint=train.fingerprint(),environment=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                   TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
    target=HERE/'reports/plan.json';target.parent.mkdir(exist_ok=True,parents=True);atomic_json(target,plan)
    return plan


def execute(plan):
    if plan['source_sha256']!=source_identity():raise ValueError('Source changed since plan preparation')
    output=protect_output(HERE/'reports/workflow')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Existing workflow; never launch duplicate experiment workers')
    output.mkdir(exist_ok=True,parents=True)
    jobs=plan['jobs'];state={j['id']:dict(status='pending',gpu=j['gpu']) for j in jobs};running={};done=set();aborted=False
    def save():
        atomic_json(output/'status.json',dict(source_sha256=plan['source_sha256'],jobs=state,aborted=aborted,
                    updated_utc=datetime.now(timezone.utc).isoformat(),workflow_pid=os.getpid()))
    try:
        while len(done)<len(jobs):
            for name,(proc,log) in list(running.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del running[name]
                state[name].update(status='completed' if code==0 else 'failed',exit_code=code)
                if code==0:done.add(name)
                else:aborted=True
                save()
            if not aborted:
                occupied={state[n]['gpu'] for n in running}
                for job in jobs:
                    name=job['id'];gpu=job['gpu']
                    if len(running)>=2 or state[name]['status']!='pending' or not set(job['depends_on'])<=done or (gpu is not None and gpu in occupied):continue
                    env=dict(os.environ)
                    for key in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
                    env.update(plan['environment']);env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
                    log=(output/f'{name}.log').open('a')
                    proc=subprocess.Popen(job['argv'],cwd=plan['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
                    running[name]=(proc,log);state[name].update(status='running',pid=proc.pid)
                    occupied.add(gpu);save()
            if aborted and not running:break
            if not running and len(done)==len(jobs):break
            time.sleep(.5)
    finally:
        for proc,log in running.values():
            if proc.poll() is None:proc.terminate()
            log.close()
    if aborted:
        for value in state.values():
            if value['status']=='pending':value['status']='not_started_after_failure'
    save()
    if aborted:raise RuntimeError('A workflow task failed; further jobs were not started')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    plan=prepare() if args.prepare else json.loads((HERE/'reports/plan.json').read_text())
    if args.execute:execute(plan)
    else:print(json.dumps({k:v for k,v in plan.items() if k!='jobs'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
