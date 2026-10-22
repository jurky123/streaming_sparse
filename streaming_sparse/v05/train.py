"""Frozen-prefix diagnostics first, then shared-prefix R2/R3 with validation selection."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import torch
from .common import (SnapshotData,BalancedSchedule,versions,DEFAULT_DATA,HERE,source_identity,
                     file_sha,protect_output,atomic_json,atomic_save,query_slices)
from .model import Router
from .primitives import statistics
from .evaluate import FrozenCache,evaluate
from .metrics import trace_summary


def configuration(phase,seed):
    protocol=json.loads((HERE/'configs/protocol.json').read_text())
    return dict(protocol[phase],phase=phase,router_seed=seed,keep=.2,
                objective='pooled SSE / pooled dense energy within each update, then equal arm mean',
                checkpoint_selection='minimum equal-arm mean validation global error; one shared historical state',
                lr=1e-4,weight_decay=.01,clip=1.,precision=protocol['precision'])


def update(model,data,chosen,optimizer,cache=None,trace=False):
    model.train();optimizer.zero_grad(set_to_none=True)
    numerators={name:[] for name in model.arms};energies=[];tiles={name:[] for name in model.arms}
    traces={};sample_rows=[];relation_traces=[]
    for version,tile in chosen:
        if cache:events,state=cache.get(version)
        else:
            events=data.load(version,'cuda');first=events[0]
            state=model.precompute(first['k_hist'],first['v_hist'],head_id=first['head_id'],layer_id=first['layer_id'])
        reference=None
        for event in events:
            selection=query_slices(event,16)[tile]
            indices=event['query_indices'][selection]
            if reference is None:reference=indices
            elif not torch.equal(reference,indices):raise ValueError('Unaligned query tiles across diffusion steps')
            q=event['q'][selection];target=event['o_dense'][selection].float()
            stat=cache.stat(version,event,tile,selection,state) if cache else statistics(q,event,state['assignment'])
            capture=trace and not traces
            routes=model.route(q,state,.2,trace=capture)
            energy=target.square().sum();energies.append(energy)
            for name,route in routes.items():
                se=(stat.output(route['mask'])-target).square().sum()
                numerators[name].append(se);tiles[name].append(float((se/(energy+1e-8)).detach()))
                if capture:
                    traces[name]=trace_summary(route,state)
                    if name=='R3':
                        for layer in route['layers']:
                            r=layer['relation']
                            if r.requires_grad:r.retain_grad();relation_traces.append(r)
            if any(not torch.isfinite(route['scores']).all() for route in routes.values()):raise FloatingPointError('Nonfinite scores')
        sample_rows.append(dict(cache_version=version,tile_id=tile,partition_hash=state['partition_hash']))
    energy=torch.stack(energies).sum()
    losses={name:torch.stack(values).sum()/(energy+1e-8) for name,values in numerators.items()}
    joint=torch.stack(list(losses.values())).mean()
    if not torch.isfinite(joint):raise FloatingPointError('Nonfinite attention loss')
    joint.backward()
    gradients={}
    if trace:
        for name,module in [('frontend',model.frontend),('shared_query_fusion',model.query_fusion),*model.branches.items()]:
            gs=[p.grad for p in module.parameters() if p.grad is not None]
            gradients[name]=sum(float(g.detach().square().sum()) for g in gs)**.5
        gradients['signed_relation']=sum(float(r.grad.square().sum()) for r in relation_traces if r.grad is not None)**.5
        gradients['correction_scale']={name:float(branch.correction_scale.grad) for name,branch in model.branches.items()
                                        if hasattr(branch,'correction_scale')}
        gradients['feature']=sum(float(p.grad.square().sum()) for p in model.frontend.kv.features.parameters() if p.grad is not None)**.5
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    optimizer.step()
    result=dict(attention_loss={k:float(v.detach()) for k,v in losses.items()},joint_loss=float(joint.detach()),
                tile_mean_relative_error={k:sum(v)/len(v) for k,v in tiles.items()},gradient_norm=norm,
                samples=sample_rows,keep=.2)
    if trace:
        result.update(representation_diagnostics=traces,gradient_diagnostics=gradients,
                      scales={name:dict(fusion=float(model.query_fusion.scale),interaction=[float(l.scale) for l in b.interaction],
                                         correction=float(b.correction_scale) if hasattr(b,'correction_scale') else None)
                              for name,b in model.branches.items()})
    return result


def run(data_root,output,phase,seed,gate,resume=None):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Expose exactly one BF16 CUDA GPU')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=configuration(phase,seed);current=source_identity()
    check=json.loads(Path(gate).read_text())
    if not check.get('passed') or check['source_sha256']!=current or not check.get('real_qkv') or not check.get('full_head_coverage'):
        raise ValueError('A current-source full36-pair real correctness gate is required')
    if check['gpu_name']!=torch.cuda.get_device_name(0):raise ValueError('Gate GPU mismatch')
    train=SnapshotData(data_root,'train');val=SnapshotData(data_root,'val')
    fingerprint=train.fingerprint()
    if fingerprint!=check['data_fingerprint']:raise ValueError('Gate data mismatch')
    train_versions=versions(train,phase,'train');val_versions=versions(val,phase,'monitor')
    random.seed(seed);torch.manual_seed(seed)
    arms=('R1','R2','R3') if phase=='phase1' else ('R2','R3')
    model=Router(arms=arms).cuda()
    if phase=='phase1':
        model.frontend.requires_grad_(False)
        # Immutable frontend reused by every decoder, every tile and all four steps.
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    schedule=BalancedSchedule(train_versions,seed)
    identity=dict(source_sha256=current,phase=phase,config=config,data_fingerprint=fingerprint,
                  train_versions=train_versions,validation_versions=val_versions,capacity=model.capacity(),
                  torch=str(torch.__version__),python=sys.version,cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    output=protect_output(output)
    if output.exists() and any(output.iterdir()) and not resume:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    begin=0;best_score=float('inf');plateau_reference=float('inf');best_step=0;stale=0;history=[]
    if resume:
        if Path(resume).resolve().parent!=output:raise ValueError('Resume in original run directory only')
        saved=torch.load(resume,map_location='cuda',weights_only=True)
        if saved['identity']!=identity:raise ValueError('Resume source/config/data/environment mismatch')
        model.load_state_dict(saved['router']);optimizer.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['sampler'])
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        random.setstate(saved['python_rng'])
        begin=saved['updates'];best_score=saved['best_score'];best_step=saved['best_step'];stale=saved['stale'];history=saved['validation_history']
        plateau_reference=saved['plateau_reference']
        metric_file=output/'metrics.jsonl'
        if metric_file.exists():
            rows=[json.loads(x) for x in metric_file.read_text().splitlines()]
            metric_file.write_text(''.join(json.dumps(x)+'\n' for x in rows if x['updates']<=begin))
    atomic_json(output/'provenance.json',identity)
    train_cache=FrozenCache(model,train,'cuda') if phase=='phase1' else None
    val_cache=FrozenCache(model,val,'cuda') if phase=='phase1' else None
    def save(path,step):
        atomic_save(dict(**model.checkpoint(),identity=identity,phase=phase,updates=step,optimizer=optimizer.state_dict(),
                        sampler=schedule.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        python_rng=random.getstate(),best_score=best_score,plateau_reference=plateau_reference,best_step=best_step,stale=stale,
                        validation_history=history),path)
    def status(label,step,**extra):
        atomic_json(output/'status.json',dict(status=label,phase=phase,updates=step,max_updates=config['max_updates'],
                    router_seed=seed,best_step=best_step,best_validation=best_score if math.isfinite(best_score) else None,
                    updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    def validate(step):
        nonlocal best_score,best_step,stale,plateau_reference
        status('validating',step)
        candidate=output/'validation_candidates'/f'update{step:06d}.pt'
        candidate.parent.mkdir(parents=True,exist_ok=True)
        save(candidate,step)
        result=evaluate(model,val,val_versions,output/f'validation/update{step:06d}',cache=val_cache,
                        checkpoint_sha=file_sha(candidate))
        metrics=result['metrics']['keep=0.2']
        values={name:metrics[name]['global_attention_error'] for name in arms}
        score=sum(values.values())/len(values)
        improved=score<best_score
        significant=score<plateau_reference*(1-config['min_relative_improvement'])
        if significant:plateau_reference=score;stale=0
        else:stale+=1
        if improved:best_score=score;best_step=step
        history.append(dict(updates=step,arms=values,joint_validation=score,best=improved,
                            significant_improvement_for_patience=significant,
                            soft={name:metrics[name+'_soft']['global_attention_error'] for name in arms},
                            mass=metrics['Mass']['global_attention_error']))
        atomic_json(output/'validation_curve.json',history)
        if improved:save(output/'best.pt',step)
        print(json.dumps(dict(event='validation',**history[-1])),flush=True)
        return result
    if not resume:
        status('initial_validation',0);save(output/'initial.pt',0);validate(0)
    final_step=begin
    for step in range(begin,config['max_updates']):
        warmup=config['warmup']
        progress=max(0.,(step-warmup)/max(1,config['max_updates']-warmup))
        multiplier=(step+1)/warmup if step<warmup else .05+.95*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:group['lr']=config['lr']*multiplier
        chosen=schedule.sample(config['snapshots_per_update'])
        row=dict(updates=step+1,lr=optimizer.param_groups[0]['lr'],**update(model,train,chosen,optimizer,train_cache,
                                                   trace=step==0 or (step+1)%50==0))
        with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        final_step=step+1;status('training',final_step)
        if step==0 or final_step%50==0:
            print(json.dumps({k:v for k,v in row.items() if k not in ('samples','representation_diagnostics')}),flush=True)
        if final_step%config['validate_every']==0:
            validate(final_step);save(output/'latest.pt',final_step)
            save(output/f'step{final_step:06d}.pt',final_step)
            if phase=='phase2' and final_step>=config['min_updates'] and stale>=config['patience']:
                print('Validation plateau stopping rule met; no convergence claim from a fixed6000 threshold.',flush=True);break
    save(output/'final.pt',final_step)
    if phase=='phase1':
        from .model import load_router
        selected,_=load_router(output/'best.pt','cuda')
        # Rebuild with the selected checkpoint to avoid stale decoder references.
        selected.frontend.requires_grad_(False)
        details=evaluate(selected,val,val_versions,output/'selected_validation_diagnostics',
                         cache=FrozenCache(selected,val,'cuda'),diagnostics=True,checkpoint_sha=file_sha(output/'best.pt'))
        initial=history[0]['joint_validation'];best=min(x['joint_validation'] for x in history)
        report=dict(passed=bool(math.isfinite(best) and best<=initial*1.25),
                    stable_only_not_architecture_success=True,initial_validation=initial,best_joint_validation=best,
                    best_step=best_step,updates=final_step,all8_training_tiles_visited=all(v>=8 for v in schedule.visits.values()),
                    unique_train_snapshots=len(schedule.visits),minimum_snapshot_visits=min(schedule.visits.values()),
                    no_test_reads=True,source_sha256=current)
        if not report['all8_training_tiles_visited']:report['passed']=False
        atomic_json(output/'phase1_result.json',report)
    else:
        atomic_json(output/'training_exposure.json',dict(final_snapshot_visits=schedule.visits,
                    final_all8_tiles_visited=all(v>=8 for v in schedule.visits.values()),
                    note='Final training exposure; selected best checkpoint has its own saved sampler visits.'))
    status('completed',final_step,stop_reason='fixed diagnostic budget' if phase=='phase1' else
           ('validation plateau' if final_step<config['max_updates'] else 'predeclared maximum; inspect validation trend before claiming convergence'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('phase1','phase2'),required=True)
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--data',default=str(DEFAULT_DATA));parser.add_argument('--gate',required=True);parser.add_argument('--resume')
    args=parser.parse_args();run(args.data,args.output,args.phase,args.seed,args.gate,args.resume)


if __name__=='__main__':main()
