"""Resumable P0 -> Pilot -> Full research workflow, no oracle supervision.

Every subprocess and stage boundary is recorded. A stopped/failed stage leaves
its data and checkpoints intact; rerun with the same protocol to resume.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HOME=Path(__file__).resolve().parent
DATA=HOME/'data'
RUNS=HOME/'runs'


def status(phase,detail,state='running'):
    value=dict(phase=phase,detail=detail,state=state,updated_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()))
    (HOME/'status.json').write_text(json.dumps(value,indent=2))
    (HOME/'STATUS.md').write_text(f"# v0.2 状态\n\n{value['updated_utc']} UTC\n\n阶段：{phase}\n\n状态：{state}\n\n{detail}\n")
    print(json.dumps(value,ensure_ascii=False),flush=True)


def command(module,args,log,gpu):
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env['PYTHONUNBUFFERED']='1'
    cmd=[sys.executable,'-m',module,*map(str,args)]
    log=Path(log)
    log.parent.mkdir(parents=True,exist_ok=True)
    with (HOME/'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(command=cmd,gpu=gpu,started=time.time(),log=str(log)))+'\n')
    with log.open('a') as f:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def create_protocol():
    path=HOME/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    texts=(ROOT/'prompts/MovieGenVideoBench_extended.txt').read_text().splitlines()[:24]
    prompts=[dict(index=i,prompt_id=hashlib.sha256(p.encode()).hexdigest()[:20],text=p,
                  split='train' if i<16 else 'val' if i<20 else 'test') for i,p in enumerate(texts)]
    assert len({r['prompt_id'] for r in prompts})==24
    value=dict(version='0.2',created=time.time(),prompts=prompts,
        pilot_indices=list(range(8))+[16,17,20,21],full_indices=list(range(24)),
        generation_seeds=[0,1],router_seeds=[0,1,2],layers=[4,14,24],heads=[0,3,6,9],
        history_chunks=[1,2,3],steps=[0,1,2,3],latent_frames=12,query_tile=16,tiles_per_event=8,
        pilot_steps=1500,full_steps=3000,snapshots_per_update=4,events_per_snapshot=4,
        training_keep=[.3,.2],evaluation_keep=[.3,.2,.1],validate_every=500,
        validation_subsample='Fixed prompt/layer/head-stratified snapshot subset; all four diffusion events and all saved Q tiles; complete final Val/Test',
        variants=['shared','conditioned','spatial','frozen','independent'],
        note='Pilot Val/Test remain in the same Full splits. No Test data used for model or hyperparameter selection. No P4 decoder enhancement enabled.')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value


def capture(protocol,indices,name,gpu):
    shared=DATA/'shared'
    shared.mkdir(parents=True,exist_ok=True)
    selected=[protocol['prompts'][i] for i in indices]
    manifest=shared/'events.jsonl'
    old=[json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
    counts=Counter(r['prompt_id'] for r in old)
    # Each prompt has 2 seeds * 12 layer/head groups * 3 history * 4 steps.
    missing=[p for p in selected if counts[p['prompt_id']]!=288]
    if missing:
        free=shutil.disk_usage(shared.resolve()).free
        estimate=len(missing)*1_200_000_000
        if free<estimate+15_000_000_000:
            raise RuntimeError(f'Insufficient disk reserve: need approx {estimate+15_000_000_000}, available {free}')
        prompts=HOME/f'{name}_capture_prompts.txt'
        prompts.write_text('\n'.join(p['text'] for p in missing)+'\n')
        status('P1 '+name,f'采集 {len(missing)} 个 Prompt × 2 Seeds × 3 Layers × 4 Heads，分层采样 8 Query Tiles。')
        command('streaming_sparse.v02.capture',['--mode','capture','--prompts',prompts,'--limit',len(missing),
            '--frames','12','--seeds','0','1','--layers','4','14','24','--heads','0','3','6','9',
            '--query-tile','16','--tiles-per-event','8','--backend','sdpa','--output',shared],HOME/f'capture_{name}.log',gpu)
    all_records=[json.loads(x) for x in manifest.read_text().splitlines()]
    ids={p['prompt_id'] for p in selected}
    rows=[r for r in all_records if r['prompt_id'] in ids]
    expected={(pid,seed,layer,head,n,step) for pid in ids for seed in (0,1) for layer in (4,14,24)
              for head in (0,3,6,9) for n in (4680,9360,14040) for step in range(4)}
    keys=[(r['prompt_id'],r['seed'],r['layer_id'],r['head_id'],r['history_tokens'],r['step_id']) for r in rows]
    if set(keys)!=expected or len(keys)!=len(expected): raise AssertionError('Incomplete or duplicate capture cells')
    versions=defaultdict(list)
    for r in rows: versions[r['cache_version']].append(r)
    if any(len(items)!=4 for items in versions.values()): raise AssertionError('Expected four Q events per history snapshot')
    for items in versions.values():
        if len({tuple(r['query_tile_starts']) for r in items})!=1:
            raise AssertionError('Diffusion steps must reuse the same Q positions')
    view=DATA/name
    view.mkdir(exist_ok=True)
    splits={s:[p['prompt_id'] for p in selected if p['split']==s] for s in ('train','val','test')}
    assert not any(set(splits[a])&set(splits[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    for r in rows:
        r['history']='../shared/'+r['history']
        r['event']='../shared/'+r['event']
    (view/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (view/'splits.json').write_text(json.dumps(splits,indent=2))
    coverage=dict(events=len(rows),snapshots=len(versions),prompts=len(ids),
        split_prompts={k:len(v) for k,v in splits.items()},
        split_events={k:sum(r['prompt_id'] in set(v) for r in rows) for k,v in splits.items()},
        max_backend_error=max(r['dense_fp32_vs_backend_error'] for r in rows),
        all_step_positions_consistent=True,complete_cross_product=True)
    if coverage['max_backend_error']>1e-4: raise AssertionError('Dense capture mismatch')
    (view/'coverage.json').write_text(json.dumps(coverage,indent=2))
    return view


def configuration(variant,seed,stage,weight,protocol):
    router=dict(conditioned=variant!='shared',grouping='frozen' if variant=='frozen' else 'spatial' if variant=='spatial' else 'learned',
                encoder_type='mlp' if variant=='independent' else 'transformer')
    return dict(seed=seed,steps=protocol[f'{stage}_steps'],lr=1e-4,weight_decay=.01,
        snapshots_per_step=4,tiles_per_event=1,keep_ratios=[.3,.2],collapse_weight=weight,
        diagnose_every=100,validate_every=500,validation_max_snapshots=24 if stage=='pilot' else 48,
        save_every=250,log_every=50,router=router)


def train(stage,variant,seed,data,weight,protocol,gpu):
    directory=RUNS/stage/variant/f'seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'training_completed.json'
    config=configuration(variant,seed,stage,weight,protocol)
    config_path=directory/'requested_config.json'
    config_path.write_text(json.dumps(config,indent=2))
    if not marker.exists():
        status('P2 '+stage,f'{variant}, Router Seed {seed}：{config["steps"]} 更新，4 snapshots/update，验证集选 best checkpoint。')
        args=['--data',data,'--config',config_path,'--output',directory]
        if (directory/'latest.pt').exists(): args+=['--resume',directory/'latest.pt']
        command('streaming_sparse.v02.train',args,directory/'train.log',gpu)
        rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
        if len(rows)!=config['steps']: raise AssertionError('Training step count mismatch')
        if not (directory/'best.pt').exists(): raise AssertionError('Missing validation-selected checkpoint')
        marker.write_text(json.dumps(dict(steps=len(rows),complete=time.time())))
    return directory


def evaluate(stage,variant,seed,directory,data,gpu,oracle=0):
    for split in ('train','val','test'):
        target=directory/f'eval_{split}'
        if (target/'summary.json').exists(): continue
        status('P3 '+stage,f'{variant}, Seed {seed}：完整 {split} 评估，B0/B1/B2/B4 和实际成本对照。')
        command('streaming_sparse.v02.evaluate',['--data',data,'--checkpoint',directory/'best.pt','--split',split,
            '--output',target,'--oracle-tiles',oracle if split in ('val','test') else 0,'--resume-existing'],directory/f'eval_{split}.log',gpu)


def report():
    command('streaming_sparse.v02.report',[],HOME/'report.log',os.environ.get('CUDA_VISIBLE_DEVICES','0'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--through',choices=['p0','pilot','full'],default='full')
    parser.add_argument('--gpus',nargs='+',type=int,help='Independent jobs across physical GPUs')
    args=parser.parse_args()
    if args.gpus:
        from .parallel_workflow import run
        try: run(args.gpus)
        except BaseException as error:
            status('parallel interrupted',str(error),'failed');raise
        return
    pid_file=HOME/'workflow.pid'
    if pid_file.exists():
        prior=int(pid_file.read_text())
        try:
            argv=(Path('/proc')/str(prior)/'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            argv=[]
        if b'streaming_sparse.v02.workflow' in argv:
            raise RuntimeError(f'Workflow already running as PID {prior}; inspect status instead of launching a duplicate')
    pid_file.write_text(str(os.getpid())+'\n')
    DATA.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    protocol=create_protocol()
    try:
        p0=RUNS/'p0'
        if not (p0/'REPORT.md').exists():
            status('P0','已有 QKV：三组 λ 对照、数值断言、分区变化与梯度贡献。')
            command('streaming_sparse.v02.p0',['--output',p0],HOME/'p0.log',args.gpu)
        p0_results=json.loads((p0/'summary.json').read_text())
        if any(not r['gradients_finite'] for r in p0_results.values()): raise AssertionError('P0 gradient gate failed')
        winner=min(p0_results.values(),key=lambda r:sum(r['errors']['val'][str(k)]['macro_prompt_relative_error'] for k in (.3,.2)))
        weight=winner['weight']
        (HOME/'selected_regularizer.json').write_text(json.dumps(dict(weight=weight,source='P0 existing validation prompts; these prompts belong to later Train split'),indent=2))
        if args.through=='p0':
            report();status('P0','P0 完成；尚未采集 Pilot。','complete');return
        pilot=capture(protocol,protocol['pilot_indices'],'pilot',args.gpu)
        for variant in ('shared','conditioned'):
            directory=train('pilot',variant,0,pilot,weight,protocol,args.gpu)
            evaluate('pilot',variant,0,directory,pilot,args.gpu,oracle=12)
        # Decide Full using training/validation stability, never Pilot Test scores.
        gate={}
        for variant in ('shared','conditioned'):
            directory=RUNS/'pilot'/variant/'seed0'
            rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
            finite=all(0<=r['attention_loss']<float('inf') and all(0<=g<float('inf') for g in r['gradients'].values()) for r in rows)
            first=sum(r['attention_loss'] for r in rows[:100])/100
            last=sum(r['attention_loss'] for r in rows[-100:])/100
            gate[variant]=dict(finite=finite,first100=first,last100=last,loss_decreased=last<first)
        (HOME/'pilot_full_gate.json').write_text(json.dumps(gate,indent=2))
        report()
        if args.through=='pilot':
            status('Pilot','Pilot 采集、A/B 联合训练与三集合评估已完成；Full 未执行。','complete');return
        if not all(r['finite'] and r['loss_decreased'] for r in gate.values()):
            status('Pilot gate','Pilot 稳定性门槛未通过；保留诊断结果，停止扩大 Full。','needs_analysis');return
        full=capture(protocol,[i for i in protocol['full_indices'] if i not in protocol['pilot_indices']],'full_additional',args.gpu)
        # The Full view includes the immutable Pilot snapshots plus the new ones.
        full=capture(protocol,protocol['full_indices'],'full',args.gpu)
        for variant in protocol['variants']:
            for seed in protocol['router_seeds']:
                directory=train('full',variant,seed,full,weight,protocol,args.gpu)
                evaluate('full',variant,seed,directory,full,args.gpu,oracle=24 if variant=='conditioned' else 0)
                report()
        status('v0.2','P0、Pilot、Full、三训练 Seed、消融和 Oracle Headroom 全部完成；P4 依据报告决定。','complete')
        report()
    except BaseException as error:
        status('interrupted',str(error),'failed')
        raise


if __name__=='__main__': main()
