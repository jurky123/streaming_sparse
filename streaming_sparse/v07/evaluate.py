"""Resumable exact hard-output evaluation for teacher diagnostics and predicted routers."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from .common import HERE,output,source_identity,atomic_json,file_sha,query_slices,pair_index,protocol
from .data import data
from .exact import teacher,current_stats,page_costs,singleton,set_gain,interaction
from .router import load,Router
from .metrics import error,summarize,sketch_metrics
from ..v05.primitives import select


def pack(mask):return np.packbits(mask.cpu().numpy().astype('uint8'),axis=-1,bitorder='little').tobytes().hex()


def row(event,version,tile,selection,keep,size,true,masks,initial=None):
    costs=page_costs(len(event['k_hist']),size,event['q'].device);budget=math.floor(keep*len(event['k_hist']))
    target=event['o_dense'][selection].float()
    r={k:event[k] for k in ('prompt_id','seed','layer_id','head_id','step_id')}
    r.update(cache_version=version,tile_id=tile,keep=keep,history_tokens=len(event['k_hist']),current_visible_length=len(event['k_current']),
        partition_type='continuous',block_size=size,num_blocks=len(costs),budget=budget,query_indices=event['query_indices'][selection].cpu().tolist(),
        errors={n:error(true.output(m),target) for n,m in masks.items()},
        retained={n:(m*costs).sum(-1).long().cpu().tolist() for n,m in masks.items() if not n.endswith('_soft')},
        gradient_guard_counts={n:true.gradient_guard_count(m) for n,m in masks.items()},changed={},jaccard={})
    if initial is not None:
        for n,m in masks.items():
            if n.endswith('_soft') or n not in ('R3-local','R4','R4-shuffle') and not n.startswith('T3'):continue
            r['changed'][n]=(m!=initial).any(-1).cpu().tolist()
            r['jaccard'][n]=((m*initial).sum(-1)/((m+initial)>0).sum(-1).clamp_min(1)).cpu().tolist()
    return r


def teacher_masks(g,costs,budget,seed=0):
    gen=torch.Generator(device=costs.device).manual_seed(seed)
    t2scores=singleton(g).clamp_min(1e-12).log()-costs.log()[None]
    t2=select(t2scores,costs,budget);gain=set_gain(g,t2)
    masks={'T0':select(torch.rand(g['mass'].shape,generator=gen,device=costs.device),costs,budget),
           'T1':select(g['mass'],costs,budget),'T2':t2}
    for beta in protocol()['teacher']['coefficients']:
        masks[f'T3_beta{beta}']=select(t2scores+beta*gain['u'].tanh(),costs,budget)
    return masks,gain


@torch.no_grad()
def evaluate(model,ds,versions,directory,keeps=(.2,),checkpoint=None,teacher_mode=False,full=False,history_only=False):
    directory=output(directory);commits=output(directory/'snapshot_commits');device='cuda' if torch.cuda.is_available() else 'cpu'
    if model is not None:device=next(model.parameters(),torch.empty(0,device=device)).device;model.eval()
    identity=dict(source_sha256=source_identity(),data_fingerprint=ds.fingerprint(),split=ds.split,snapshots=sorted(versions),
                  keeps=list(keeps),checkpoint_sha256=checkpoint,teacher_diagnostic=teacher_mode,full=full,history_only=history_only,
                  stat_source='teacher_diagnostic' if teacher_mode else 'predicted',page_size=model.page_size if hasattr(model,'page_size') else model.sketch.page_size if model else 64)
    if (directory/'identity.json').exists() and json.loads((directory/'identity.json').read_text())!=identity:raise ValueError('Evaluation identity changed')
    atomic_json(directory/'identity.json',identity);allrows=[];statrows=[];diag=[]
    size=identity['page_size']
    for number,version in enumerate(sorted(versions)):
        target=commits/f'{version}.json'
        if target.exists():
            saved=json.loads(target.read_text())
            if saved['source_sha256']!=identity['source_sha256'] or len(saved['rows'])!=32*len(keeps):raise ValueError('Bad snapshot commit')
        else:
            events=ds.load(version,device);first=events[0];pair=pair_index(first['layer_id'],first['head_id'])
            sketch=None if teacher_mode else model if hasattr(model,'prototypes') else model.sketch
            state=sketch(first['k_hist'],first['v_hist'],pair) if sketch else None
            rows=[];stats=[];details=[]
            for event in events:
                for tile,sl in enumerate(query_slices(event,16)):
                    q=event['q'][sl];true=teacher(q,event,size);current=(true.current_logz,true.current_output)
                    if sketch:
                        pred=sketch.estimate(q,state,current)
                        sm=sketch_metrics(pred,true)
                        pa=singleton(pred).clamp_min(1e-12).log();ta=singleton(true.geometry()).clamp_min(1e-12).log()
                        pa=pa-pa.mean(-1,keepdim=True);ta=ta-ta.mean(-1,keepdim=True)
                        sm['singleton_logscore_pearson']=float(((pa*ta).sum(-1)/(pa.norm(dim=-1)*ta.norm(dim=-1)).clamp_min(1e-12)).mean())
                        sm.update(prompt_id=event['prompt_id'],seed=event['seed'],layer_id=event['layer_id'],head_id=event['head_id'],
                            cache_version=version,step_id=event['step_id'],tile_id=tile);stats.append(sm)
                    for keep in keeps:
                        costs=page_costs(len(event['k_hist']),size,q.device);budget=math.floor(keep*len(event['k_hist']))
                        if teacher_mode:
                            masks,gain=teacher_masks(true.geometry(),costs,budget,seed=20261018+number*32+event['step_id']*8+tile)
                            initial=masks['T2']
                        elif hasattr(model,'prototypes'):
                            score=singleton(pred).clamp_min(1e-12).log()-costs.log()[None]
                            masks={'SketchSingleton':select(score,costs,budget),'PredictedMass':select(pred['mass'],costs,budget),
                                   'TrueSingleton':select(singleton(true.geometry()).clamp_min(1e-12).log()-costs.log()[None],costs,budget)}
                            initial=None
                        else:
                            routes=model.route(q,state,current,keep,history_only=history_only)
                            masks={n:r['hard_mask'] for n,r in routes.items()}
                            masks.update({n+'_soft':r['soft_mask'] for n,r in routes.items() if n!='R1'})
                            if full and not history_only:
                                hist=model.route(q,state,current,keep,history_only=True)
                                masks['R2-history-reference']=hist['R2']['hard_mask']
                                if 'R4' in model.arms:masks['R4-shuffle']=model.route(q,state,current,keep,shuffle=True)['R4']['hard_mask']
                            initial=masks['R2'] if history_only else masks['R3']
                        masks['Mass']=select(true.geometry()['mass'],costs,budget)
                        r=row(event,version,tile,sl,keep,size,true,masks,initial)
                        if not teacher_mode and not hasattr(model,'prototypes'):
                            r['correction_diagnostics']={}
                            for name,route in routes.items():
                                if 'delta' not in route:continue
                                item=dict(alpha=float(route['alpha']),delta_score_norm=route['delta'].norm(dim=-1).cpu().tolist(),
                                          base_score_norm=route['initial_scores'].norm(dim=-1).cpu().tolist())
                                condition=route.get('set_condition')
                                if condition is not None:
                                    item.update(initial_estimated_error=condition['error'].cpu().tolist(),
                                        retained_predicted_mass=condition['z'].cpu().tolist(),
                                        gain_abs_max=condition['gain'].abs().amax(-1).cpu().tolist(),
                                        normalized_gain_abs_max=condition['u'].abs().amax(-1).cpu().tolist())
                                r['correction_diagnostics'][name]=item
                        if 'SketchSingleton' in masks:
                            pm,tm=masks['SketchSingleton'],masks['TrueSingleton']
                            r['sketch_mask_overlap']=((pm*tm).sum(-1)/((pm+tm)>0).sum(-1).clamp_min(1)).cpu().tolist()
                        ig=interaction(true.geometry(),initial if initial is not None else masks['Mass'])
                        r['interaction']={k:v.cpu().tolist() for k,v in ig.items()}
                        if full:r['masks_hex']={n:pack(m) for n,m in masks.items() if not n.endswith('_soft')}
                        rows.append(r)
            saved=dict(source_sha256=source_identity(),rows=rows,statistics=stats,diagnostics=details,
                       block_sizes=page_costs(len(first['k_hist']),size,device).int().cpu().tolist())
            atomic_json(target,saved)
        allrows.extend(saved['rows']);statrows.extend(saved['statistics']);diag.extend(saved['diagnostics'])
        atomic_json(directory/'progress.json',dict(completed=number+1,total=len(versions),split=ds.split))
    result=dict(identity=identity,metrics=summarize(allrows),sketch_statistics=statistics_summary(statrows) if statrows else None)
    atomic_json(directory/'summary.json',result)
    return result


def statistics_summary(rows):
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:
        groups['all'].append(r)
        for key in ('prompt_id','seed','layer_id','head_id','step_id'):groups[f'{key}={r[key]}'].append(r)
        groups[f"pair={r['layer_id']}:{r['head_id']}"] .append(r)
    result={}
    for key,values in groups.items():
        scalars={k:sum(r[k] for r in values)/len(values) for k in
         ('logz_mae','logz_mse','weighted_logz_abs','mass_l1','current_mass_abs','weighted_value_error','high_mass_value_error','residual_norm_abs_bias')}
        scalars['logz_rmse']=math.sqrt(scalars['logz_mse'])
        count=sum(r['logz_count'] for r in values)
        scalars['block_logz_mae']=sum(r['logz_abs_sum'] for r in values)/count
        scalars['block_logz_rmse']=math.sqrt(sum(r['logz_sq_sum'] for r in values)/count)
        mass=sum(r['logz_mass_sum'] for r in values)
        scalars['mass_weighted_logz_mae']=sum(r['logz_mass_abs_sum'] for r in values)/max(mass,1e-30)
        scalars['mass_weighted_logz_rmse']=math.sqrt(sum(r['logz_mass_sq_sum'] for r in values)/max(mass,1e-30))
        scalars['singleton_logscore_pearson']=sum(r['singleton_logscore_pearson'] for r in values)/len(values)
        scalars['residual_direction_cosine']=sum(r['residual_cosine_sum'] for r in values)/max(1,sum(r['residual_valid_count'] for r in values))
        scalars['residual_low_norm_count']=sum(r['residual_low_norm_count'] for r in values)
        scalars['dense_micro_error']=sum(r['dense_error']['sse'] for r in values)/(sum(r['dense_error']['energy'] for r in values)+1e-8)
        video=defaultdict(list)
        for r in values:video[(r['prompt_id'],r['seed'])].append(r)
        dense_groups=[sum(r['dense_error']['sse'] for r in rs)/(sum(r['dense_error']['energy'] for r in rs)+1e-8) for rs in video.values()]
        scalars['dense_macro_error']=sum(dense_groups)/len(dense_groups)
        result[key]=scalars
    return result


def compose(directory,device):
    directory=Path(directory);base,saved=load(directory/'initial.pt',device)
    for name in base.arms:
        other,value=load(directory/f'best_{name}.pt',device)
        if other.base_digest()!=base.base_digest():raise ValueError('Corrections must share the exact same base/sketch')
        base.corrections[name].load_state_dict(other.corrections[name].state_dict())
    base.freeze_base();return base,{n:file_sha(directory/f'best_{n}.pt') for n in base.arms}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--compose');p.add_argument('--teacher',action='store_true')
    p.add_argument('--split',choices=('train','val'),required=True);p.add_argument('--output',required=True);p.add_argument('--monitor',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ds=data(a.split)
    if a.teacher:model=None;sha=None
    elif a.compose:model,sha=compose(a.compose,'cuda')
    else:model,saved=load(a.checkpoint,'cuda');sha=file_sha(a.checkpoint)
    from .common import monitor_versions,train_monitor_versions
    selected=(monitor_versions(ds) if ds.split=='val' else train_monitor_versions(ds)) if a.monitor else sorted(ds.groups)
    evaluate(model,ds,selected,a.output,(.2,.1,.3),sha,a.teacher,full=True,
             history_only=not a.teacher and not a.compose and saved.get('identity',{}).get('mode')=='R2')


if __name__=='__main__':main()
