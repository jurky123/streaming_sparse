"""Two-GPU dependency scheduler. Default is dry-run; pair branches stay colocated."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .identity import source_identity
from .paths import protect_output
from ..v02.evaluation_journal import atomic_json


def validate(plan):
    jobs=plan['execution']['commands']
    identifiers=[j['id'] for j in jobs]
    if len(identifiers)!=len(set(identifiers)):raise ValueError('Duplicate job IDs')
    known=set(identifiers)
    for job in jobs:
        if set(job['depends_on'])-known:raise ValueError('Unknown dependency')
        gpu=job['gpu_physical_index']
        if gpu not in (None,0,1):raise ValueError('Only physical GPUs0/1 are configured')
        visible=job['environment'].get('CUDA_VISIBLE_DEVICES')
        if visible!=(str(gpu) if gpu is not None else ''):raise ValueError('Job has ambiguous GPU visibility')
        if not job['id'] or any(c in job['id'] for c in ('/','\\')):raise ValueError('Invalid job ID')
    resolved=set()
    while len(resolved)<len(jobs):
        ready={j['id'] for j in jobs if set(j['depends_on'])<=resolved}-resolved
        if not ready:raise ValueError('Dependency cycle')
        resolved|=ready


def execute(plan,output):
    validate(plan)
    if plan['source_sha256']!=source_identity():raise ValueError('Regenerate plan for current source')
    output=protect_output(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Workflow directory already used; never reuse old PIDs or launch duplicate workers')
    output.mkdir(parents=True,exist_ok=True)
    execution=plan['execution']
    jobs=execution['commands']
    status={j['id']:dict(status='pending',gpu_physical_index=j['gpu_physical_index']) for j in jobs}
    running={}
    completed=set()
    aborted=False
    def save():
        atomic_json(output/'status.json',dict(source_sha256=plan['source_sha256'],jobs=status,
            max_simultaneous_gpu_jobs=2,max_jobs_per_gpu=1,aborted=aborted))
    try:
        while len(completed)<len(jobs):
            for identifier,(proc,stream) in list(running.items()):
                code=proc.poll()
                if code is None:continue
                stream.close();del running[identifier]
                status[identifier].update(status='completed' if code==0 else 'failed',exit_code=code)
                if code==0:completed.add(identifier)
                else:aborted=True
                save()
            if not aborted:
                busy={status[k]['gpu_physical_index'] for k in running}
                for job in jobs:
                    identifier=job['id'];gpu=job['gpu_physical_index']
                    if (len(running)>=2 or status[identifier]['status']!='pending' or
                        not set(job['depends_on'])<=completed or (gpu is not None and gpu in busy)):
                        continue
                    env=dict(os.environ)
                    for key in execution.get('clear_proxy_variables',[]):env.pop(key,None)
                    env.update(execution.get('environment',{}));env.update(job['environment'])
                    stream=(output/f'{identifier}.log').open('a')
                    proc=subprocess.Popen(job['argv'],cwd=execution['cwd'],env=env,
                                          stdout=stream,stderr=subprocess.STDOUT)
                    running[identifier]=(proc,stream)
                    status[identifier].update(status='running',pid=proc.pid)
                    if gpu is not None:busy.add(gpu)
                    save()
            if aborted and not running:break
            if not running and len(completed)==len(jobs):break
            time.sleep(.2)
    finally:
        # Only owned child processes are ever stopped; PID files are not reused.
        for proc,stream in running.values():
            if proc.poll() is None:proc.terminate()
            stream.close()
    if aborted:
        for value in status.values():
            if value['status']=='pending':value['status']='not_started_after_failure'
        save();raise RuntimeError('Workflow job failed; inspect per-job logs, no further jobs launched')
    save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default=str(Path(__file__).parent/'reports/preparation.json'))
    p.add_argument('--output',default=str(Path(__file__).parent/'reports/workflow'))
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    plan=json.loads(Path(args.plan).read_text());validate(plan)
    if not args.execute:
        print(json.dumps(dict(status='dry_run_no_jobs_started',jobs=len(plan['execution']['commands']),
            physical_gpus=[0,1],max_jobs_per_gpu=1,source_sha256=plan['source_sha256']),indent=2))
        return
    execute(plan,args.output)


if __name__=='__main__':main()
