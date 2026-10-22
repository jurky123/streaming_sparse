"""GPU-aware persistent DAG; respect existing workers, no proxy and fail closed."""
import argparse,csv,json,os,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from .common import HERE,ROOT,source_identity,atomic_json,protocol,output,file_sha,train_monitor_versions
from .data import data,audit
from .progress import render


def prepare():
    py=str(HERE.parent/'.venv/bin/python');jobs=[]
    def add(name,module,args,depends=(),gpu=True,condition=None,directory=None):
        jobs.append(dict(id=name,argv=[py,'-u','-m',module,*map(str,args)],depends=list(depends),gpu_required=gpu,
                         condition=condition,directory=str(directory) if directory else None))
    add('unit_correctness','unittest',['streaming_sparse.v07.tests','-v'],gpu=False)
    add('phase0_real36','streaming_sparse.v07.sanity',[],['unit_correctness'])
    add('phase1_teacher_val','streaming_sparse.v07.evaluate',['--teacher','--split','val','--output',HERE/'phase1/full_val'],
        ['phase0_real36'],directory=HERE/'phase1/full_val')
    add('phase1_teacher_decision','streaming_sparse.v07.report',['--teacher-decision'],['phase1_teacher_val'],gpu=False)
    add('phase1_swap_probe','streaming_sparse.v07.diagnostics',[],['phase1_teacher_decision'],directory=HERE/'phase1/fixed_swap_probe')
    add('S1_control','streaming_sparse.v07.train',['--phase','sketch','--variant','S1','--seed',0],['phase1_teacher_decision'],directory=HERE/'runs/seed0/S1')
    add('S1_full_val','streaming_sparse.v07.evaluate',['--checkpoint',HERE/'runs/seed0/S1/best_sketch.pt','--split','val','--output',HERE/'runs/seed0/S1/full_val'],['S1_control'],directory=HERE/'runs/seed0/S1/full_val')
    for seed in (0,1,2):
        root=HERE/f'runs/seed{seed}'
        for variant in ('S4','S8'):
            task=f'{variant}_seed{seed}';directory=root/variant
            add(task,'streaming_sparse.v07.train',['--phase','sketch','--variant',variant,'--seed',seed],['phase1_teacher_decision'],directory=directory)
            add(task+'_full_val','streaming_sparse.v07.evaluate',['--checkpoint',directory/'best_sketch.pt','--split','val','--output',directory/'full_val'],[task],directory=directory/'full_val')
    add('sketch_sufficiency_decision','streaming_sparse.v07.report',['--sketch-decision'],
        ['S1_full_val',*[f'S4_seed{s}_full_val' for s in (0,1,2)]],gpu=False)
    terminals=['phase1_swap_probe',*[f'S8_seed{s}_full_val' for s in (0,1,2)]]
    for seed in (0,1,2):
        root=HERE/f'runs/seed{seed}'
        for phase,modelname in (('base','R3'),('R2','R2')):
            task=f'{modelname}_seed{seed}';directory=root/modelname
            add(task,'streaming_sparse.v07.train',['--phase',phase,'--seed',seed],['sketch_sufficiency_decision'],directory=directory)
            for split,flag in (('val',[]),('train',['--monitor'])):
                name=task+'_'+split
                add(name,'streaming_sparse.v07.evaluate',['--checkpoint',directory/f'best_{modelname}.pt','--split',split,
                    '--output',directory/f'{"full" if split=="val" else "monitor"}_{split}',*flag],[task],directory=directory/f'{"full" if split=="val" else "monitor"}_{split}')
                terminals.append(name)
        task=f'correction_seed{seed}';directory=root/'correction'
        add(task,'streaming_sparse.v07.train',['--phase','correction','--seed',seed],[f'R3_seed{seed}','sketch_sufficiency_decision'],directory=directory)
        for split,flag in (('val',[]),('train',['--monitor'])):
            name=task+'_'+split
            add(name,'streaming_sparse.v07.evaluate',['--compose',directory,'--split',split,'--output',directory/f'{"full" if split=="val" else "monitor"}_{split}',*flag],
                [task],directory=directory/f'{"full" if split=="val" else "monitor"}_{split}');terminals.append(name)
        for variant in ('S2','S4-var','S4-K'):
            task=f'secondary_{variant}_seed{seed}';directory=root/variant;condition='sketch_insufficient'
            add(task,'streaming_sparse.v07.train',['--phase','sketch','--variant',variant,'--seed',seed],['sketch_sufficiency_decision'],condition=condition,directory=directory)
            name=task+'_full_val'
            add(name,'streaming_sparse.v07.evaluate',['--checkpoint',directory/'best_sketch.pt','--split','val','--output',directory/'full_val'],[task],condition=condition,directory=directory/'full_val');terminals.append(name)
    add('final_train_val_report','streaming_sparse.v07.report',[],terminals,gpu=False)
    train=data('train');val=data('val');plan=dict(source_sha256=source_identity(),cwd=str(ROOT),jobs=jobs,protocol=protocol(),
        data_fingerprint=train.fingerprint(),train_snapshots=len(train.groups),val_snapshots=len(val.groups),train_monitor=train_monitor_versions(train),
        hook_audit=audit(),env=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR=str(HERE/'reports/mpl_cache')))
    path=HERE/'reports/plan.json';output(path.parent)
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Existing different frozen plan')
    atomic_json(path,plan);return plan


def gpu_state():
    a=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    b=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    index={row[1].strip():int(row[0]) for row in csv.reader(a.stdout.splitlines())};busy={i:[] for i in index.values()}
    for row in csv.reader(b.stdout.splitlines()):
        if len(row)>=2 and row[0].strip() in index:busy[index[row[0].strip()]].append(dict(pid=row[1].strip(),process=row[2].strip() if len(row)>2 else ''))
    return busy


def allowed(condition):
    if condition is None:return True
    decision=json.loads((HERE/'reports/sketch_decision.json').read_text())
    if decision['source_sha256']!=source_identity():raise ValueError('Stale decision source')
    return decision['train_secondary_sketches']


def execute(plan):
    if plan['source_sha256']!=source_identity():raise ValueError('Source changed after preparation')
    path=HERE/'reports/workflow'
    if path.exists() and any(path.iterdir()):raise FileExistsError('Refuse duplicate workflow launch')
    output(path);snap=output(HERE/'reports/launched_source')
    for f in HERE.glob('*.py'):shutil.copy2(f,snap/f.name)
    shutil.copy2(HERE/'configs/protocol.json',snap/'protocol.json')
    jobs=plan['jobs'];states={j['id']:dict(status='pending',gpu=None,directory=j['directory']) for j in jobs}
    running={};done=set();aborted=False;occupancy={};last_poll=0;last_save=0
    def save():
        atomic_json(path/'status.json',dict(source_sha256=plan['source_sha256'],jobs=states,aborted=aborted,
                   workflow_pid=os.getpid(),gpu_occupancy=occupancy,updated_utc=datetime.now(timezone.utc).isoformat()))
        (HERE/'reports/STATUS.txt').write_text(render())
    save()
    try:
        while len(done)<len(jobs):
            for name,(proc,log) in list(running.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del running[name]
                states[name].update(status='completed' if code==0 else 'failed',exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat())
                if code==0:done.add(name)
                else:aborted=True
                save()
            if not aborted:
                if time.monotonic()-last_poll>5:
                    occupancy=gpu_state();last_poll=time.monotonic()
                occupied={states[n]['gpu'] for n in running if states[n]['gpu'] is not None}
                for job in jobs:
                    name=job['id']
                    if states[name]['status']!='pending' or not set(job['depends'])<=done:continue
                    if not allowed(job['condition']):
                        states[name].update(status='skipped',reason='Predeclared full-Val S4 sufficiency condition does not require secondary variants');done.add(name);save();continue
                    if len(running)>=3:continue
                    gpu=None
                    if job['gpu_required']:
                        gpu=next((i for i in (0,1) if i in occupancy and not occupancy[i] and i not in occupied),None)
                        if gpu is None:continue
                    env=dict(os.environ)
                    for key in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
                    env.update(plan['env']);env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
                    log=(path/f'{name}.log').open('a');proc=subprocess.Popen(job['argv'],cwd=plan['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
                    running[name]=(proc,log);states[name].update(status='running',gpu=gpu,pid=proc.pid,started_utc=datetime.now(timezone.utc).isoformat())
                    if gpu is not None:occupied.add(gpu)
                    save()
            if aborted and not running:break
            if not running and len(done)==len(jobs):break
            # Waiting for an old GPU worker is a valid state, not a DAG deadlock.
            if time.monotonic()-last_save>15:save();last_save=time.monotonic()
            time.sleep(1)
    finally:
        for name,(proc,log) in running.items():
            if proc.poll() is None:proc.terminate();states[name].update(status='failed',reason='Workflow interrupted')
            log.close()
        save()
    if aborted:raise RuntimeError('Task failed; no further jobs dispatched')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    plan=prepare() if a.prepare else json.loads((HERE/'reports/plan.json').read_text())
    if a.execute:execute(plan)
    else:print(json.dumps({k:v for k,v in plan.items() if k not in ('jobs','protocol','hook_audit','train_monitor')},indent=2))


if __name__=='__main__':main()
