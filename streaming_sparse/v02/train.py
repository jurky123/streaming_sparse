"""Balanced multi-snapshot router training with separate gradient/discrete diagnostics."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time
import torch
from .model import RouterConfig,SparseRouter
from .data import SnapshotData
from .metrics import cluster_diagnostics,partition_change,error_parts
from .evaluate import evaluate,statistics
from ..capture.dataset_io import atomic_save,query_slices
from ..attention.block_statistics import exact_attention
from ..evaluation.baselines import mass_selection
from ..train.train_router import gradient_norms


def gradient_vector(loss,parameters):
    if not loss.requires_grad or not parameters:
        return None
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(parameters,gradients)])


def probe(model,sample,selection,keep,old_partition=None):
    with torch.no_grad():
        q=sample['q'][selection]
        target=sample['o_dense'][selection]
        state=model.precompute(sample['k_hist'],sample['v_hist'],sample['token_positions'],sample['head_id'],sample['layer_id'],old_partition)
        route=model.route(q,state,keep)
        stats=statistics(q,sample,state['assignment'])
        output=stats.output(route['hard_mask'])
        se,energy=error_parts(output,target)
        mass=mass_selection(stats,state['hard'].sum(0),route['budget'])
        mass_se,_=error_parts(stats.output(mass),target)
        result=dict(hard_error=float(se/(energy+1e-8)),mass_error=float(mass_se/(energy+1e-8)),
                    kept_tokens=int((state['hard'].sum(0)*route['hard_mask']).sum()),
                    mass_tokens=int((state['hard'].sum(0)*mass).sum()),budget=route['budget'])
        return result,state


def run(config,data_root,output,device='cuda',resume=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.jsonl').exists() and resume is None:
        raise FileExistsError('Use a fresh experiment directory or --resume')
    seed=config.get('seed',0)
    rng=random.Random(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    model=SparseRouter(RouterConfig(**config.get('router',{}))).to(device)
    data=SnapshotData(data_root,'train')
    val=SnapshotData(data_root,'val') if config.get('validate_every',0) else None
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config.get('lr',1e-4),weight_decay=config.get('weight_decay',.01))
    begin,best=0,float('inf')
    if resume:
        saved=torch.load(resume,weights_only=True,map_location=device)
        if saved['config']!=asdict(model.config): raise ValueError('Resume config mismatch')
        model.load_state_dict(saved['router'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        begin,best=saved['step']+1,saved['best_val']
        # Discard trace rows beyond the last durable optimizer state when
        # resuming an interrupted write/validation interval.
        for name,limit in [('metrics.jsonl',begin-1),('validation_curve.jsonl',begin)]:
            path=output/name
            if path.exists():
                # An external signal can interrupt the final JSONL append.
                rows=[json.loads(x) for x in path.read_text().splitlines(keepends=True) if x.endswith('\n')]
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=limit))
    (output/'config.json').write_text(json.dumps(config,indent=2))
    if begin==0 and not resume:
        atomic_save(model.checkpoint(),output/'initial.pt')
        atomic_save({**model.checkpoint(),'optimizer':optimizer.state_dict(),'step':-1,'best_val':best,
                     'python_rng':rng.getstate(),'torch_rng':torch.get_rng_state()},output/'latest.pt')
    group_parameters=[p for p in model.features.group.parameters() if p.requires_grad]
    steps=config.get('steps',1500)
    weight=config.get('collapse_weight',1e-3)
    for step in range(begin,steps):
        tick=time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        versions=data.sample_versions(rng,config.get('snapshots_per_step',4))
        keep=rng.choice(config.get('keep_ratios',[.3,.2]))
        diagnose=step%config.get('diagnose_every',50)==0 or step==steps-1
        attention_total,regularizer_total,sse_total,energy_total=0.,0.,0.,0.
        diagnostics=[]
        event_rows=[]
        grad_attention=grad_regularizer=None
        probe_data=None
        for si,version in enumerate(versions):
            events=data.load(version,device)
            sample=events[0]
            state=model.precompute(sample['k_hist'],sample['v_hist'],sample['token_positions'],sample['head_id'],sample['layer_id'])
            if not torch.equal(state['hard'].sum(-1),torch.ones(len(sample['k_hist']),device=device)):
                raise AssertionError('Token assignment must be one-hot')
            attention=0
            for ei,event in enumerate(events):
                available=query_slices(event,model.config.query_tile)
                selections=rng.sample(available,min(len(available),config.get('tiles_per_event',1)))
                event_se,event_energy=0,0
                event_kept=[]
                for selection in selections:
                    q=event['q'][selection]
                    route=model.route(q,state,keep)
                    stats=statistics(q,event,state['assignment'])
                    result=stats.output(route['mask'])
                    se,energy=error_parts(result,event['o_dense'][selection])
                    event_se=event_se+se
                    event_energy=event_energy+energy
                    kept=int((state['hard'].sum(0)*route['hard_mask']).sum())
                    event_kept.append(kept)
                    if kept>route['budget']: raise AssertionError('Budget exceeded')
                    if diagnose and si==0 and ei==0:
                        with torch.no_grad():
                            soft_mask_se,_=error_parts(stats.output(route['soft_mask']),event['o_dense'][selection])
                            soft_stats=statistics(q,event,state['probabilities'].detach())
                            soft_cluster_se,_=error_parts(soft_stats.output(route['hard_mask']),event['o_dense'][selection])
                            selected=state['hard']@route['hard_mask']>0
                            direct=exact_attention(q,torch.cat((event['k_hist'][selected],event['k_current'])),
                                                   torch.cat((event['v_hist'][selected],event['v_current'])))
                            torch.testing.assert_close(result.detach(),direct,atol=5e-5,rtol=5e-5)
                            torch.testing.assert_close(stats.dense(),event['o_dense'][selection].float(),atol=5e-5,rtol=5e-5)
                            labels=state['hard'].argmax(-1).detach()
                            diagnostics.append(dict(**cluster_diagnostics(state),
                                cache_version=version,layer=event['layer_id'],head=event['head_id'],
                                soft_mask_error=float(soft_mask_se/(energy+1e-8)),
                                soft_clustering_fixed_mask_error=float(soft_cluster_se/(energy+1e-8)),
                                hard_soft_mask_gap=float((se.detach()-soft_mask_se)/(energy+1e-8)),
                                hard_soft_clustering_gap=float((se.detach()-soft_cluster_se)/(energy+1e-8)),
                                actual_keep=kept/len(event['k_hist']),kept_tokens=kept,budget=route['budget']))
                            probe_data=(event,selection,labels,state['hard'].detach().clone())
                            atomic_save(dict(labels=labels.cpu(),raw_labels=state['raw_labels'].cpu() if state['raw_labels'] is not None else None,
                                selected_blocks=torch.where(route['hard_mask']>0)[0].cpu(),query_indices=event['query_indices'][selection].cpu(),
                                cache_version=version,step=step),output/'assignments'/f'step{step:06d}.pt')
                attention=attention+event_se/(event_energy+1e-8)/len(events)
                sse_total+=float(event_se.detach())
                energy_total+=float(event_energy)
                event_rows.append(dict(prompt_id=event['prompt_id'],seed=event['seed'],layer_id=event['layer_id'],head_id=event['head_id'],
                    step_id=event['step_id'],history_tokens=len(event['k_hist']),cache_version=version,
                    squared_error=float(event_se.detach()),dense_energy=float(event_energy),
                    normalized_error=float(event_se.detach()/(event_energy+1e-8)),
                    actual_keep=sum(event_kept)/len(event_kept)/len(event['k_hist'])))
            regularizer=state['regularizer']
            if diagnose:
                ga=gradient_vector(attention/len(versions),group_parameters)
                gr=gradient_vector(regularizer/len(versions),group_parameters)
                if ga is not None: grad_attention=ga if grad_attention is None else grad_attention+ga
                if gr is not None: grad_regularizer=gr if grad_regularizer is None else grad_regularizer+gr
            loss=(attention+weight*regularizer)/len(versions)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            attention_total+=float(attention.detach())/len(versions)
            regularizer_total+=float(regularizer.detach())/len(versions)
        norms=gradient_norms(model)
        if any(not math_finite(v) for v in norms.values()): raise FloatingPointError('Nonfinite module gradient')
        if step==0 and config.get('require_gradients',True):
            required=['content','aggregator','encoder','decoder']
            if model.config.grouping=='learned': required.append('grouping')
            if any(norms[k]<=0 for k in required): raise AssertionError(f'Missing gradient: {norms}')
        torch.nn.utils.clip_grad_norm_(model.parameters(),config.get('grad_clip',1.),error_if_nonfinite=True)
        optimizer.step()
        row=dict(step=step,attention_loss=attention_total,cluster_kl=regularizer_total,
            weighted_cluster_loss=weight*regularizer_total,cluster_loss_ratio=weight*regularizer_total/max(attention_total,1e-12),
            global_relative_error=sse_total/max(energy_total,1e-8),squared_error=sse_total,dense_energy=energy_total,
            gradients=norms,keep=keep,snapshots=versions,cluster_diagnostics=diagnostics)
        row['events']=event_rows
        if diagnose and grad_attention is not None and grad_regularizer is not None:
            na,nr=float(grad_attention.norm()),float(grad_regularizer.norm())
            row.update(group_attention_gradient=na,group_regularizer_gradient=nr,
                weighted_regularizer_gradient_ratio=weight*nr/max(na,1e-12),
                group_gradient_cosine=float(grad_attention@grad_regularizer)/(max(na*nr,1e-20)))
        if diagnose and probe_data:
            event,selection,old_labels,old_hard=probe_data
            new_result,new_state=probe(model,event,selection,keep)
            old_result,_=probe(model,event,selection,keep,old_hard)
            row['partition_update']={**partition_change(old_labels,new_state['hard'].argmax(-1)),
                'new_partition_error':new_result['hard_error'],'old_partition_error_at_same_postupdate_parameters':old_result['hard_error'],
                'partition_only_error_change':new_result['hard_error']-old_result['hard_error'],
                'new_partition_mass_error':new_result['mass_error'],'old_partition_mass_error':old_result['mass_error'],
                'new_kept_tokens':new_result['kept_tokens'],'old_kept_tokens':old_result['kept_tokens']}
        row['elapsed_ms']=(time.perf_counter()-tick)*1000
        with (output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if step%config.get('log_every',50)==0:
            print(json.dumps({k:v for k,v in row.items() if k not in ('snapshots','cluster_diagnostics','events')}),flush=True)
        if val is not None and ((step+1)%config['validate_every']==0 or step==steps-1):
            validation_path=output/'validation'/f'step{step+1:06d}'
            arguments=dict(keeps=(.3,.2),baselines=False,max_snapshots=config.get('validation_max_snapshots'),save_masks=False,bootstrap=200)
            try:
                report=evaluate(model,val,validation_path,**arguments)
            except ValueError as error:
                if not str(error).startswith(('Evaluation resume identity mismatch','Legacy partial evaluation exists')):
                    raise
                # Replaying updates after the last optimizer checkpoint may
                # produce different parameters. Preserve the old validation,
                # then evaluate the replayed model without mixing records.
                archive=output/'validation_replayed'/f'step{step+1:06d}_{time.time_ns()}'
                archive.parent.mkdir(exist_ok=True)
                shutil.move(str(validation_path),str(archive))
                print(json.dumps(dict(event='validation_replay_archived',archive=str(archive),reason=str(error))),flush=True)
                report=evaluate(model,val,validation_path,**arguments)
            score=sum(report['metrics'][f'Ours/keep={k}']['macro_prompt_head_relative_error'] for k in (.3,.2))/2
            with (output/'validation_curve.jsonl').open('a') as f: f.write(json.dumps(dict(step=step+1,score=score))+'\n')
            if score<best:
                best=score
                atomic_save({**model.checkpoint(),'selected_step':step+1,'selection_score':score},output/'best.pt')
        if (step+1)%config.get('save_every',250)==0 or step==steps-1:
            atomic_save({**model.checkpoint(),'optimizer':optimizer.state_dict(),'step':step,'best_val':best,
                         'python_rng':rng.getstate(),'torch_rng':torch.get_rng_state()},output/'latest.pt')
            atomic_save(model.checkpoint(),output/'final.pt')
    return model


def math_finite(value):
    import math
    return math.isfinite(value)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume')
    args=p.parse_args()
    run(json.loads(Path(args.config).read_text()),args.data,args.output,args.device,args.resume)


if __name__=='__main__': main()
