"""Sketch pretraining -> frozen-Sketch relevance -> identical-base corrections."""
import argparse
from datetime import datetime,timezone
import json,math,os,random
from pathlib import Path
import torch
from .common import (HERE,protocol,source_identity,output,atomic_json,atomic_save,file_sha,pair_index,
                     BalancedSchedule,monitor_versions,query_slices,tensor_digest)
from .data import data
from .exact import teacher,current_stats
from .sketch import Sketch
from .router import Router,load
from .metrics import sketch_losses
from .evaluate import evaluate

VARIANTS={'S1':dict(prototypes=1),'S2':dict(prototypes=2),'S4':dict(prototypes=4),'S8':dict(prototypes=8),
          'S4-var':dict(prototypes=4,variance=True),'S4-K':dict(prototypes=4,key_only=True)}


def initial_scales(seed,train):
    target=HERE/f'runs/seed{seed}/initial_sketch_scales.json'
    if target.exists():return json.loads(target.read_text())
    torch.manual_seed(seed);model=Sketch(4).cuda();values={k:[] for k in ('Z','V','O')}
    # Train metadata only: one snapshot per pair, all4 steps, one fixed tile.
    with torch.no_grad():
        for i,(_,vs) in enumerate(sorted(train.strata.items())):
            events=train.load(min(vs),'cuda');first=events[0];pair=pair_index(first['layer_id'],first['head_id'])
            state=model(first['k_hist'],first['v_hist'],pair)
            for event in events:
                q=event['q'][query_slices(event,16)[i%8]];true=teacher(q,event)
                pred=model.estimate(q,state,(true.current_logz,true.current_output))
                for k,v in sketch_losses(pred,true).items():values[k].append(float(v))
    means={k:sum(v)/len(v) for k,v in values.items()};raw={k:1/max(v,1e-8) for k,v in means.items()}
    scale=sum(raw.values());weights={k:min(100.,3*v/scale) for k,v in raw.items()}
    result=dict(source_sha256=source_identity(),train_fingerprint=train.fingerprint(),seed=seed,means=means,weights=weights,
                samples=values,no_validation_or_test_used=True)
    output(target.parent);atomic_json(target,result);return result


def update(model,train,samples,optimizer,mode,weights,active):
    optimizer.zero_grad(set_to_none=True);numerators={n:[] for n in active};energies=[];raw={k:[] for k in ('Z','V','O')};metadata=[]
    for version,tile in samples:
        events=train.load(version,'cuda');first=events[0];pair=pair_index(first['layer_id'],first['head_id'])
        sk=model if mode=='sketch' else model.sketch
        state=sk(first['k_hist'],first['v_hist'],pair);old_indices=None
        for event in events:
            sl=query_slices(event,16)[tile];q=event['q'][sl];target=event['o_dense'][sl].float()
            indices=event['query_indices'][sl]
            if old_indices is None:old_indices=indices
            elif not torch.equal(old_indices,indices):raise ValueError('Unaligned step query windows')
            true=teacher(q,event,sk.page_size)
            if mode=='sketch':
                pred=model.estimate(q,state,(true.current_logz,true.current_output))
                for k,v in sketch_losses(pred,true).items():raw[k].append(v)
            else:
                current=current_stats(q,event['k_current'],event['v_current'])
                routes=model.route(q,state,current,.2,history_only=mode=='R2')
                energies.append(target.square().sum())
                for name in active:
                    route=routes[name]
                    if not torch.isfinite(route['scores']).all():raise FloatingPointError('Nonfinite router scores')
                    # Teacher constants are only supplied to the loss, after mask selection.
                    numerators[name].append((true.output(route['mask'])-target).square().sum())
        metadata.append(dict(cache_version=version,tile_id=tile,partition_hash=state['partition_hash']))
    if mode=='sketch':
        losses={k:torch.stack(v).mean() for k,v in raw.items()}
        objective=sum(weights[k]*v for k,v in losses.items())
    else:
        energy=torch.stack(energies).sum()
        losses={n:torch.stack(v).sum()/(energy+1e-8) for n,v in numerators.items()}
        objective=sum(losses.values())
    if not torch.isfinite(objective):raise FloatingPointError('Nonfinite loss')
    objective.backward()
    if mode=='correction':
        if any(p.grad is not None for module in (model.sketch,model.base) for p in module.parameters()):raise AssertionError('Frozen base received gradients')
        norms={n:float(torch.nn.utils.clip_grad_norm_(model.corrections[n].parameters(),1.,error_if_nonfinite=True)) for n in active}
    else:norms={'all':float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))}
    optimizer.step()
    result=dict(loss={k:float(v.detach()) for k,v in losses.items()},gradient_norm_before_clip=norms,samples=metadata,
                objective=float(objective.detach()),keep=.2)
    if mode=='correction':result['alpha']={n:torch.nn.functional.softplus(model.corrections[n].alpha_raw).detach().cpu().tolist() for n in model.arms}
    return result


def run(phase,seed,variant=None,resume=None,page_size=64,kind=None):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one CUDA GPU')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=protocol();gate=json.loads((HERE/'reports/phase0.json').read_text())
    if not gate['passed'] or gate['source_sha256']!=source_identity():raise ValueError('Current-source Phase0 must PASS before training')
    train=data('train');val=data('val')
    if train.fingerprint()!=gate['data_fingerprint']:raise ValueError('Raw data changed after Phase0 gate')
    weights=initial_scales(seed,train)['weights']
    torch.manual_seed(seed);random.seed(seed);root=HERE/f'runs/seed{seed}'
    prefix='' if page_size==64 else f'page{page_size}_'
    if phase=='sketch':
        if variant not in VARIANTS:raise ValueError(variant)
        model=Sketch(**VARIANTS[variant],page_size=page_size).cuda();mode='sketch';active=['sketch'];directory=root/f'{prefix}{variant}'
    elif phase in ('base','R2'):
        sk,saved=load(root/f'{prefix}S4/best_sketch.pt','cuda')
        model=Router(sk.config()).cuda();model.sketch.load_state_dict(sk.state_dict());model.sketch.requires_grad_(False);model.sketch_frozen=True
        mode='R2' if phase=='R2' else 'base';active=['R2' if phase=='R2' else 'R3'];directory=root/f'{prefix}{active[0]}'
    elif phase=='correction':
        base,saved=load(root/f'{prefix}R3/best_R3.pt','cuda')
        decision=json.loads((HERE/'reports/teacher_decision.json').read_text())
        sufficient=json.loads((HERE/'reports/sketch_decision.json').read_text())['sufficient']
        names=('R3-local','R4') if decision['headroom'] and sufficient else ('R3-local',)
        model=Router(base.sketch.config(),names).cuda();own=model.state_dict();own.update(base.state_dict());model.load_state_dict(own);model.freeze_base()
        mode='correction';active=list(names);directory=root/f'{prefix}correction'
    elif phase in ('joint','scratch'):
        # Optional explicitly separate training-condition configs; never mixed into frozen_same_base.
        if phase=='joint':
            base,saved=load(root/f'{prefix}correction/best_R4.pt','cuda')
            model=Router(base.sketch.config(),('R4',)).cuda();own=model.state_dict()
            own.update({k:v for k,v in base.state_dict().items() if k in own});model.load_state_dict(own)
        else:model=Router(dict(prototypes=4,page_size=page_size),('R4',)).cuda()
        mode='joint';active=['R4'];directory=root/f'{prefix}{phase}_R4'
    else:raise ValueError(phase)
    directory=output(directory)
    if any(directory.iterdir()) and not resume:raise FileExistsError(directory)
    settings=dict(p['sketch' if phase=='sketch' else 'correction' if phase=='correction' else 'joint' if phase=='joint' else 'base'])
    frozen=model.base_digest() if mode=='correction' else None
    identity=dict(source_sha256=source_identity(),phase=phase,mode=mode,seed=seed,variant=variant,page_size=page_size,config=settings,
                  dataset_fingerprint=train.fingerprint(),initial_base_digest=frozen,precision=p['precision'],weights=weights,
                  torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  parameters=sum(v.numel() for v in model.parameters()),trainable=sum(v.numel() for v in model.parameters() if v.requires_grad))
    parameters=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=settings['lr'],weight_decay=.01) if parameters else None
    schedule=BalancedSchedule(sorted(train.groups),seed);begin=0
    best={n:float('inf') for n in active};reference=dict(best);stale={n:0 for n in active};best_step={n:0 for n in active};history=[];arm_updates={n:0 for n in active}
    def checkpoint(step):
        value=dict(kind='sketch',config=model.config(),state=model.state_dict()) if mode=='sketch' else dict(kind='router',**model.checkpoint())
        return dict(**value,identity=identity,updates=step,optimizer=optimizer.state_dict() if optimizer else None,
                    sampler=schedule.state_dict(),best=best,plateau_reference=reference,stale=stale,best_step=best_step,active=active,
                    arm_updates=arm_updates,validation_history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate())
    if resume:
        s=torch.load(resume,map_location='cuda',weights_only=True)
        if s['identity']!=identity:raise ValueError('Resume identity mismatch')
        model.load_state_dict(s['state'] if mode=='sketch' else s['router']);optimizer.load_state_dict(s['optimizer']);schedule.load_state_dict(s['sampler'])
        begin=s['updates'];best=s['best'];reference=s['plateau_reference'];stale=s['stale'];best_step=s['best_step'];history=s['validation_history'];active=s['active'];arm_updates=s['arm_updates']
        torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']]);random.setstate(s['python_rng'])
        mf=directory/'metrics.jsonl'
        if mf.exists():mf.write_text(''.join(json.dumps(x)+'\n' for x in map(json.loads,mf.read_text().splitlines()) if x['updates']<=begin))
    atomic_json(directory/'provenance.json',identity)
    def status(label,step,**extra):
        atomic_json(directory/'status.json',dict(status=label,phase=phase,seed=seed,variant=variant,updates=step,max_updates=settings['max_updates'],
                    best={k:v if math.isfinite(v) else None for k,v in best.items()},best_step=best_step,active=active,arm_updates=arm_updates,
                    updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    def validate(step):
        if frozen and model.base_digest()!=frozen:raise AssertionError('Frozen base drift')
        status('validating',step);candidate=directory/'validation_candidates'/f'update{step:06d}.pt';atomic_save(checkpoint(step),candidate)
        result=evaluate(model,val,monitor_versions(val),directory/f'validation/update{step:06d}',checkpoint=file_sha(candidate),history_only=mode=='R2')
        metrics=result['metrics']['keep=0.2']
        if mode=='sketch':scores={'sketch':result['sketch_statistics']['all']['dense_micro_error']}
        elif mode=='R2':
            # R2 needs independent historical-only route; its dedicated validation below
            # explicitly evaluates that input restriction, rather than default R3 scores.
            scores={'R2':metrics['R2']['micro_error']}
        else:scores={n:metrics[n]['micro_error'] for n in active}
        improved=[]
        for n,score in scores.items():
            if score<reference[n]*(1-settings['min_relative_improvement']):reference[n]=score;stale[n]=0
            else:stale[n]+=1
            if score<best[n]:best[n]=score;best_step[n]=step;improved.append(n)
        history.append(dict(updates=step,scores=scores,improved=improved,mass=metrics['Mass']['micro_error'],
                            base=metrics.get('R3',{}).get('micro_error'),frozen_digest=model.base_digest() if frozen else None))
        atomic_json(directory/'validation_curve.json',history)
        for n in improved:atomic_save(checkpoint(step),directory/f'best_{n}.pt')
        print(json.dumps(dict(event='validation',**history[-1])),flush=True)
    if not resume:status('initializing',0);atomic_save(checkpoint(0),directory/'initial.pt');validate(0)
    final=begin
    if optimizer:
        for step in range(begin,settings['max_updates']):
            warm=settings['warmup'];progress=max(0.,(step-warm)/max(1,settings['max_updates']-warm))
            scale=(step+1)/warm if step<warm else .05+.95*.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:group['lr']=settings['lr']*scale
            values=update(model,train,schedule.sample(settings['snapshots_per_update']),optimizer,mode,weights,active)
            row=dict(updates=step+1,lr=optimizer.param_groups[0]['lr'],**values)
            with (directory/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            final=step+1
            for n in active:arm_updates[n]=final
            status('training',final)
            if final==1 or final%100==0:print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
            if final%settings['validate_every']==0 or final==settings['max_updates']:
                validate(final)
                if final>=settings['min_updates']:active=[n for n in active if stale[n]<settings['patience']]
                atomic_save(checkpoint(final),directory/'latest.pt')
                if not active:break
    atomic_save(checkpoint(final),directory/'final.pt')
    status('completed',final,stop_reason='Deterministic no-parameter S1 mean' if not optimizer else
           'Validation plateau' if final<settings['max_updates'] else 'Predeclared maximum; no convergence claim')
    atomic_json(directory/'exposure.json',dict(visits=schedule.visits,all8_tiles=all(v>=8 for v in schedule.visits.values()) if optimizer else None,
                 selected_exposure='Each selected checkpoint contains its own sampler, distinct from final exposure'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('sketch','base','R2','correction','joint','scratch'),required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True);p.add_argument('--variant',choices=tuple(VARIANTS));p.add_argument('--resume')
    p.add_argument('--page-size',type=int,choices=(64,128),default=64)
    a=p.parse_args();run(a.phase,a.seed,a.variant,a.resume,a.page_size)


if __name__=='__main__':main()
