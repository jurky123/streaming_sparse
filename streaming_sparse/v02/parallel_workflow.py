"""Single controller with bounded evaluation concurrency and durable DAG resume."""
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from . import workflow as serial
from .evaluation_journal import atomic_json

HOME=Path(__file__).resolve().parent
ROOT=HOME.parents[1]
SCRATCH=Path(os.environ.get('STREAMING_SPARSE_SCRATCH',str(HOME/'runtime')))


def atomic_text(path,text):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def gpu_memory():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
    return {int(a):int(b) for a,b in (line.split(',') for line in result.stdout.splitlines())}


class Pool:
    def __init__(self,gpus):
        self.gpus=gpus;self.active={};self.pending=[];self.done=[];self.failed=[]
        policy=HOME/'configs/evaluation_concurrency.json'
        self.evaluation_slots=int(json.loads(policy.read_text()).get('slots_per_gpu',1)) if policy.exists() else 1
        if self.evaluation_slots not in (1,2):raise ValueError('Evaluation slots must be 1 or 2')
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        self.uuid_index={uuid.strip():int(index) for index,uuid in (row.split(',') for row in rows.splitlines())}

    def add(self,identifier,module,args,log,kind,priority=10,dependencies=(),allowed=None,**extra):
        self.pending.append(dict(id=identifier,module=module,args=list(map(str,args)),log=str(log),
            kind=kind,priority=priority,dependencies=list(dependencies),allowed=allowed,
            attempts=0,ready_at=0,**extra))

    def snapshot(self,phase):
        payload=dict(updated_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),pid=os.getpid(),
            phase=phase,gpus=self.gpus,active=[{k:v for k,v in j.items() if k not in ('process','stream')} for j in self.active.values()],
            pending=[{k:v for k,v in j.items()} for j in self.pending],completed=self.done,failed=self.failed)
        atomic_json(HOME/'parallel_status.json',payload)
        counts=Counter(j['kind'] for j in self.done)
        detail=f"GPUs {self.gpus}; completed {dict(counts)}; active " + ', '.join(f"GPU{j['gpu']}/slot{j['slot']}:{j['id']}" for j in self.active.values())
        waiting=bool(self.pending) and not self.active and not self.failed
        serial.status(phase,detail,'waiting_resources' if waiting else 'running')

    def start(self,job,gpu,slot=0):
        job['attempts']+=1;job['gpu']=gpu;job['slot']=slot;job['started']=time.time()
        if job['kind']=='evaluation':
            output=Path(job['args'][job['args'].index('--output')+1]);progress=output/'progress.json'
            job['initial_snapshots']=json.loads(progress.read_text()).get('snapshots_done',0) if progress.exists() else 0
        args=list(job['args'])
        if job['kind']=='capture':
            directory=Path(job['output_base'])/f"attempt{job['attempts']}"
            if directory.exists():
                directory=directory.with_name(directory.name+'_'+str(time.time_ns()))
            directory.mkdir(parents=True)
            args+=['--output',str(directory)];job['output']=str(directory)
        elif job['kind']=='train':
            latest=Path(job['directory'])/'latest.pt'
            if latest.exists():args+=['--resume',str(latest)]
        command=[sys.executable,'-u','-m',job['module'],*args]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        path=Path(job['log']);path.parent.mkdir(parents=True,exist_ok=True)
        stream=path.open('a')
        stream.write('\n'+json.dumps(dict(event='parallel_start',job=job['id'],gpu=gpu,attempt=job['attempts'],command=command))+'\n');stream.flush()
        job['stream']=stream
        job['process']=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                        stdout=stream,stderr=subprocess.STDOUT,close_fds=True)
        job['pid']=job['process'].pid
        self.active[gpu,slot]=job
        with (HOME/'logs/parallel_commands.jsonl').open('a') as f:
            f.write(json.dumps(dict(job=job['id'],gpu=gpu,command=command,started=job['started'],pid=job['pid']))+'\n')

    def run(self,phase,on_complete=None):
        last=0
        while self.pending or self.active:
            if (HOME/'supervisor.stop').exists():raise KeyboardInterrupt('Explicit supervisor stop marker')
            changed=False
            for slot_key,job in list(self.active.items()):
                gpu=job['gpu']
                code=job['process'].poll()
                if code is None:continue
                job['stream'].close();del self.active[slot_key]
                clean={k:v for k,v in job.items() if k not in ('process','stream')}
                clean.update(returncode=code,finished=time.time());changed=True
                with (HOME/'logs/parallel_commands.jsonl').open('a') as f:
                    f.write(json.dumps(dict(job=job['id'],gpu=gpu,returncode=code,finished=clean['finished']))+'\n')
                if code==0:
                    if on_complete:on_complete(clean,self)
                    self.done.append(clean)
                elif code<0 and job['attempts']<3:
                    clean['ready_at']=time.time()+30
                    self.pending.append(clean)
                else:self.failed.append(clean)
            completed={j['id'] for j in self.done}
            blocked={j['id'] for j in self.failed}
            for job in list(self.pending):
                if any(d in blocked for d in job['dependencies']):
                    self.pending.remove(job);self.failed.append({**job,'error':'dependency failed'});changed=True
            memory=gpu_memory() if self.pending else {}
            app_counts=Counter()
            if self.pending and self.evaluation_slots>1:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                    '--format=csv,noheader'],text=True)
                app_counts.update(self.uuid_index.get(row.strip(),-1) for row in rows.splitlines())
            for job in sorted(list(self.pending),key=lambda j:(j['priority'],j['id'])):
                if job['ready_at']>time.time() or not set(job['dependencies'])<=completed:continue
                candidates=[]
                for gpu in self.gpus:
                    if job['allowed'] is not None and gpu not in job['allowed']:continue
                    residents=[j for j in self.active.values() if j['gpu']==gpu]
                    if not residents and memory.get(gpu,999999)<512:
                        candidates.append((gpu,0));continue
                    if (job['kind']!='evaluation' or self.evaluation_slots!=2 or len(residents)!=1
                        or residents[0]['kind']!='evaluation' or memory.get(gpu,999999)>=8192
                        or app_counts[gpu]!=1):continue
                    # A freshly committed CUDA snapshot proves that the sole GPU
                    # context belongs to the resident job, even when NVML PIDs
                    # are reported in a different PID namespace. Other contexts
                    # cause a conservative refusal to add a second evaluator.
                    resident=residents[0]
                    output=Path(resident['args'][resident['args'].index('--output')+1])
                    progress=output/'progress.json'
                    try:advanced=json.loads(progress.read_text()).get('snapshots_done',0)>resident['initial_snapshots']
                    except (OSError,ValueError):advanced=False
                    if advanced:candidates.append((gpu,1-resident['slot']))
                if not candidates:continue
                # Spread across idle GPUs before sharing an already owned GPU.
                gpu,slot=min(candidates,key=lambda c:(sum(j['gpu']==c[0] for j in self.active.values()),c[0]))
                self.pending.remove(job);self.start(job,gpu,slot);changed=True
            if changed or time.time()-last>30:
                self.snapshot(phase);last=time.time()
            if self.pending and not self.active and self.failed:
                unresolved=[j for j in self.pending if not set(j['dependencies'])<=completed]
                if unresolved and len(unresolved)==len(self.pending):break
            time.sleep(2)
        if self.failed or self.pending:raise RuntimeError(f'Parallel tasks failed: {[j["id"] for j in self.failed]}')


def merge_capture(jobs,protocol):
    shared=HOME/'data/shared'
    old=[json.loads(x) for x in (shared/'events.jsonl').read_text().splitlines()]
    existing={(r['prompt_id'],r['seed'],r['layer_id'],r['head_id'],r['history_tokens'],r['step_id']) for r in old}
    for job in jobs:
        if job['kind']!='capture':continue
        source=Path(job['output'])
        rows=[json.loads(x) for x in (source/'events.jsonl').read_text().splitlines()]
        expected_prompts={protocol['prompts'][i]['prompt_id'] for i in job['prompt_indices']}
        assert len(rows)==len(expected_prompts)*288
        assert {r['prompt_id'] for r in rows}==expected_prompts
        counts=Counter(r['prompt_id'] for r in rows);assert set(counts.values())=={288}
        for r in rows:
            key=(r['prompt_id'],r['seed'],r['layer_id'],r['head_id'],r['history_tokens'],r['step_id'])
            if key in existing:continue
            for field in ('history','event'):
                src=source/r[field];dest=shared/r[field];dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    if hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(dest.read_bytes()).digest():
                        raise ValueError(f'Capture file collision: {dest}')
                else:os.link(src,dest)
            existing.add(key);old.append(r)
    atomic_text(shared/'events.jsonl',''.join(json.dumps(r)+'\n' for r in old))
    return serial.capture(protocol,protocol['full_indices'],'full',7)


def add_evaluations(pool,variant,seed,directory,data,dependency=None):
    for split in ('val','test','train'):
        output=directory/f'eval_{split}'
        if (output/'summary.json').exists():continue
        oracle=24 if variant=='conditioned' and split!='train' else 0
        pool.add(f'eval_{variant}_s{seed}_{split}','streaming_sparse.v02.evaluate',
            ['--data',data,'--checkpoint',directory/'best.pt','--split',split,'--output',output,
             '--oracle-tiles',oracle,'--resume-existing'],directory/f'eval_{split}.log','evaluation',
             priority={'val':5,'test':6,'train':20}[split],dependencies=[dependency] if dependency else [])


def run(gpus):
    if len(gpus)!=len(set(gpus)):raise ValueError('Duplicate GPU IDs')
    with (HOME/'logs/parallel.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid=HOME/'workflow.pid'
        if pid.exists():
            prior=Path('/proc')/pid.read_text().strip()/'cmdline'
            if prior.exists() and b'streaming_sparse.v02.workflow' in prior.read_bytes().split(b'\0'):
                raise RuntimeError('Existing workflow is active')
        pid.write_text(str(os.getpid())+'\n')
        protocol=serial.create_protocol()
        gate={}
        for variant in ('shared','conditioned'):
            directory=HOME/'runs/pilot'/variant/'seed0'
            if not (directory/'training_completed.json').exists():raise RuntimeError('Pilot training must complete first')
            rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
            finite=all(0<=r['attention_loss']<float('inf') and all(0<=v<float('inf') for v in r['gradients'].values()) for r in rows)
            first=sum(r['attention_loss'] for r in rows[:100])/100;last=sum(r['attention_loss'] for r in rows[-100:])/100
            gate[variant]=dict(finite=finite,first100=first,last100=last,loss_decreased=last<first)
        atomic_json(HOME/'pilot_full_gate.json',gate)
        if not all(v['finite'] and v['loss_decreased'] for v in gate.values()):
            serial.status('Pilot gate','Pilot stability gate failed','needs_analysis');return
        weight=json.loads((HOME/'selected_regularizer.json').read_text())['weight']
        initial=Pool(gpus)
        for variant in ('shared','conditioned'):
            directory=HOME/'runs/pilot'/variant/'seed0'
            for split in ('train','val','test'):
                output=directory/f'eval_{split}'
                if (output/'summary.json').exists():continue
                initial.add(f'pilot_{variant}_{split}','streaming_sparse.v02.evaluate',
                    ['--data',HOME/'data/pilot','--checkpoint',directory/'best.pt','--split',split,
                     '--output',output,'--oracle-tiles',12 if split!='train' else 0,'--resume-existing'],
                    directory/f'eval_{split}.log','pilot_evaluation',priority=0,allowed=[7] if 7 in gpus else None)
        manifest=HOME/'data/shared/events.jsonl'
        counts=Counter(json.loads(x)['prompt_id'] for x in manifest.read_text().splitlines())
        missing=[i for i in protocol['full_indices'] if counts[protocol['prompts'][i]['prompt_id']]==0]
        if any(counts[protocol['prompts'][i]['prompt_id']] not in (0,288) for i in protocol['full_indices']):
            raise RuntimeError('Partial shared capture requires quarantine before merging')
        for shard in range(min(4,len(missing),len(gpus))):
            indices=missing[shard::min(4,len(missing),len(gpus))]
            prompts=HOME/'configs'/f'full_capture_shard{shard}.txt'
            prompts.write_text('\n'.join(protocol['prompts'][i]['text'] for i in indices)+'\n')
            initial.add(f'capture_full_{shard}','streaming_sparse.v02.capture',
                ['--mode','capture','--prompts',prompts,'--limit',len(indices),'--frames',12,'--seeds',0,1,
                 '--layers',4,14,24,'--heads',0,3,6,9,'--query-tile',16,'--tiles-per-event',8,'--backend','sdpa'],
                HOME/'logs'/f'capture_full_shard{shard}.log','capture',priority=1,
                output_base=str(SCRATCH/'data/capture_shards'/f'full_{shard}'),prompt_indices=indices)
        initial.run('P1 Full parallel capture / Pilot completion')
        full=merge_capture(initial.done,protocol)
        serial.report()
        pool=Pool(gpus)
        # First cover all variants and the three primary Conditioned seeds.
        pairs=[('conditioned',s) for s in protocol['router_seeds']]
        pairs += [(v,s) for s in protocol['router_seeds'] for v in protocol['variants'] if v!='conditioned']
        for index,(variant,seed) in enumerate(pairs):
            directory=HOME/'runs/full'/variant/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            config=serial.configuration(variant,seed,'full',weight,protocol)
            requested=directory/'requested_config.json'
            if requested.exists() and json.loads(requested.read_text())!=config:raise ValueError('Run config changed')
            atomic_json(requested,config)
            marker=directory/'training_completed.json'
            identifier=f'train_{variant}_s{seed}'
            if not marker.exists():
                pool.add(identifier,'streaming_sparse.v02.train',
                    ['--data',full,'--config',requested,'--output',directory],directory/'train.log','train',
                    priority=10+index/100,directory=str(directory),variant=variant,seed=seed,steps=config['steps'])
            add_evaluations(pool,variant,seed,directory,full,None if marker.exists() else identifier)
        def complete(job,pool):
            if job['kind']=='train':
                directory=Path(job['directory'])
                rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
                if len(rows)!=job['steps'] or not (directory/'best.pt').exists() or not (directory/'final.pt').exists():
                    raise AssertionError('Training completion audit failed')
                atomic_json(directory/'training_completed.json',dict(steps=len(rows),complete=time.time()))
        pool.run('P2/P3 Full multi-GPU',complete)
        serial.report()
        serial.status('Review packaging','All experiments finished; auditing and packaging review artifacts')
        serial.command('streaming_sparse.v02.review_pack',[],HOME/'logs/review_pack.log',gpus[0])
        serial.status('v0.2','Full 15 training runs, 45 evaluations and review packages complete','complete')
