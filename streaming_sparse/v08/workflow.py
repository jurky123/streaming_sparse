"""Explicit two-GPU stages; failures stop dispatch and Test stays sealed."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import Counter
from .common import HERE,ROOT,protocol,source_identity,atomic_json,output


def plan():
    p=protocol();jobs=[]
    def add(name,stage,module,args=(),dependencies=(),gpu=True):
        jobs.append(dict(name=name,stage=stage,module='streaming_sparse.v08.'+module,args=list(args),
                         dependencies=list(dependencies),gpu=gpu))
        return name
    unit=add('unit_and_tiny_wan_audit','prepare','audit',['--tiny-network'])
    train=add('capture_train','prepare','collect',['--split','train'],[unit])
    val=add('capture_val','prepare','collect',['--split','val'],[unit])
    real=add('real_network_and_raw_QKV_audit','prepare','audit',['--real'],[train,unit])
    mains=[]
    for seed in p['training_seeds']:
        cal=add(f'calibration_seed{seed}','pilot' if seed==0 else 'matrix','warmstart',['--seed',str(seed)],[real])
        heads=[]
        for kind in ('K','V'):
            heads.append(add(f'heads_{kind}_seed{seed}','pilot' if seed==0 else 'matrix','train',
                ['--stage','heads','--kind',kind,'--seed',str(seed)],[cal,train,val]))
        if seed==0:
            pilot=[add(f'pilot_{group}_seed0','pilot','train',['--stage','main','--group',group,'--seed','0','--pilot'],heads)
                   for group in p['core_pilot']]
        core=[]
        for group in p['core_pilot']:
            deps=heads+pilot if seed==0 else heads+seed0_done
            core.append(add(f'main_{group}_seed{seed}','matrix','train',
                ['--stage','main','--group',group,'--seed',str(seed)],deps))
        controls=[add(f'main_{group}_seed{seed}','matrix','train',['--stage','main','--group',group,'--seed',str(seed)],heads+core)
                  for group in ('A-K','A-V')]
        mains+=core+controls
        if seed==0:seed0_done=core+controls
    lock=add('lock_all_checkpoints','final','evaluate',['--make-lock',str(HERE/'reports/final_lock.json')],mains,gpu=False)
    eval_jobs=[]
    for seed in p['training_seeds']:
        for group in p['groups']:
            for tag in ('initial','best','last'):
                path=HERE/f'runs/seed{seed}/{group}/{tag}.pt'
                target=HERE/f'evaluations/val/seed{seed}/{group}/{tag}'
                eval_jobs.append(add(f'val_{group}_seed{seed}_{tag}','final','evaluate',
                    ['--checkpoint',str(path),'--split','val','--output',str(target),'--mass-reference','--mechanisms'],[lock]))
    test_capture=add('capture_new_sealed_test','test','collect',['--split','test'],[lock])
    for seed in p['training_seeds']:
        for group in p['groups']:
            for tag in ('initial','best','last'):
                path=HERE/f'runs/seed{seed}/{group}/{tag}.pt';target=HERE/f'evaluations/test/seed{seed}/{group}/{tag}'
                add(f'test_{group}_seed{seed}_{tag}','test','evaluate',
                    ['--checkpoint',str(path),'--split','test','--lock',str(HERE/'reports/final_lock.json'),
                     '--output',str(target),'--mass-reference','--mechanisms'],[test_capture])
            add(f'rollout_{group}_seed{seed}','test','rollout',
                ['--checkpoint',str(HERE/f'runs/seed{seed}/{group}/best.pt'),'--split','test',
                 '--lock',str(HERE/'reports/final_lock.json'),'--output',str(HERE/f'rollout/seed{seed}/{group}'),
                 '--teacher-on-student'],[test_capture])
    for split,stage in [('val','final'),('test','test')]:
        dependencies=eval_jobs if split=='val' else [j['name'] for j in jobs if j['name'].startswith('test_')]
        add(f'paired_factorial_report_{split}',stage,'report',['--split',split],dependencies,gpu=False)
    return dict(source_sha256=source_identity(),stages=['prepare','pilot','matrix','final','test'],jobs=jobs,
        no_automatic_H0_R1_set_extensions=True,test_only_after_explicit_through_test=True)


def gpu_busy():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={uuid.strip():int(index) for index,uuid in (r.split(',') for r in devices.splitlines() if r.strip())}
    return {mapping[line.split(',')[0].strip()] for line in result.splitlines() if line.strip()}


def status_text(state):
    counts=Counter(v['status'] for v in state['jobs'].values())
    lines=['CAVIR v0.8：可重放网络输出蒸馏 / 局部 LoRA 共适应',
        f"范围截至 {state['through']}；完成{counts['completed']} / 运行{counts['running']} / 等待{counts['pending']} / 失败{counts['failed']}",
        'main为正式2000 updates；pilot为临时完整输入小样本接口检查；Test默认封存。']
    for name,v in state['jobs'].items():
        if v['status']=='running':lines.append(f"{name} | GPU {v.get('gpu')} | PID {v.get('pid')} | 日志 {v['log']}")
    if state.get('failure'):lines.append('停止派发：'+state['failure'])
    (HERE/'reports/STATUS.txt').write_text('\n'.join(lines)+'\n')


def run(through):
    frozen=plan();directory=output(HERE/'reports/workflow');path=directory/'status.json'
    if path.exists():
        state=json.loads(path.read_text())
        if state['source_sha256']!=source_identity():raise ValueError('Workflow source changed; preserve previous attempt before rebuilding')
        if any(v['status']=='running' for v in state['jobs'].values()):raise RuntimeError('Prior running jobs exist; audit actual processes, never reuse old PIDs')
        if any(v['status']=='failed' for v in state['jobs'].values()):raise RuntimeError('A previous task failed; inspect its log and resume its own checkpoint before advancing this workflow')
    else:state=dict(source_sha256=source_identity(),jobs={})
    included=set(frozen['stages'][:frozen['stages'].index(through)+1])
    jobs=[j for j in frozen['jobs'] if j['stage'] in included]
    state['through']=through;state.pop('failure',None)
    atomic_json(HERE/'reports/plan.json',frozen)
    launch=output(HERE/'reports/launched_source')
    for f in HERE.glob('*.py'):shutil.copyfile(f,launch/f.name)
    shutil.copyfile(HERE/'configs/protocol.json',launch/'protocol.json')
    for j in jobs:state['jobs'].setdefault(j['name'],dict(status='pending'))
    children={};logs={}
    while True:
        if source_identity()!=frozen['source_sha256']:raise RuntimeError('Experiment source edited while workflow active')
        for name,(child,gpu) in list(children.items()):
            code=child.poll()
            if code is not None:
                state['jobs'][name].update(status='completed' if code==0 else 'failed',exit_code=code)
                logs.pop(name).close();del children[name]
                if code:state['failure']=f'{name}退出码{code}；检查日志，不自动更改协议或重启'
        if not state.get('failure'):
            busy=gpu_busy()|{g for _,g in children.values() if g is not None}
            for j in jobs:
                entry=state['jobs'][j['name']]
                if entry['status']!='pending':continue
                if not all(state['jobs'].get(d,{}).get('status')=='completed' for d in j['dependencies']):continue
                available=[i for i in (0,1) if i not in busy]
                if j['gpu'] and not available:continue
                gpu=available[0] if j['gpu'] else None
                env=os.environ.copy()
                for key in list(env):
                    if key.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(key,None)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4')
                env.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
                logfile=directory/(j['name']+'.log');log=logfile.open('a');logs[j['name']]=log
                child=subprocess.Popen([sys.executable,'-u','-m',j['module'],*j['args']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[j['name']]=(child,gpu);entry.update(status='running',pid=child.pid,gpu=gpu,log=str(logfile))
                if gpu is not None:busy.add(gpu)
        atomic_json(path,state);status_text(state)
        if not children and (state.get('failure') or all(state['jobs'][j['name']]['status']=='completed' for j in jobs)):break
        if not children and all(state['jobs'][j['name']]['status']!='pending' for j in jobs):break
        time.sleep(2)
    if state.get('failure'):raise RuntimeError(state['failure'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--status',action='store_true')
    p.add_argument('--through',choices=['prepare','pilot','matrix','final','test'],default='pilot')
    a=p.parse_args()
    if a.status:
        path=HERE/'reports/STATUS.txt';print(path.read_text() if path.exists() else '尚未启动正式 v0.8 队列');return
    if a.run:run(a.through)
    else:
        x=plan();atomic_json(HERE/'reports/plan.json',x)
        print(json.dumps(dict(stages=x['stages'],jobs_by_stage=dict(Counter(j['stage'] for j in x['jobs'])),
            source_sha256=x['source_sha256'],default_through='pilot',execution='add --run explicitly'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
