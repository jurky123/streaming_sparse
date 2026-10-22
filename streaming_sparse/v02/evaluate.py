import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import torch
from .model import load_router
from .data import SnapshotData
from .metrics import aggregate,cluster_diagnostics,error_parts
from .oracle import forward_greedy,refine_mass,batch_error
from .evaluation_journal import EvaluationJournal,identity,atomic_json
from ..capture.dataset_io import query_slices,atomic_save
from ..attention.block_statistics import differentiable_block_stats
from ..evaluation.baselines import baseline_assignment,mass_selection
from ..router.budget_selection import select_under_budget


def statistics(q,sample,assignment):
    return differentiable_block_stats(q,sample['k_hist'],sample['v_hist'],assignment,
                                     sample['k_current'],sample['v_current'])


@torch.no_grad()
def evaluate(model,data,output=None,keeps=(.3,.2,.1),baselines=True,oracle_tiles=0,
             max_snapshots=None,save_masks=True,bootstrap=1000,resume_existing=False):
    model.eval()
    device=next(model.parameters()).device
    versions=sorted(data.groups)
    if max_snapshots:
        # Spread the diagnostic subset across layer/head groups, not first hash IDs.
        versions=[]
        strata=defaultdict(list)
        for version,items in data.groups.items():
            r=items[0]
            strata[(r['layer_id'],r['head_id'],r['prompt_id'])].append(version)
        for offset in range(max(len(v) for v in strata.values())):
            for group_index,(key,items) in enumerate(sorted(strata.items())):
                if offset<len(items): versions.append(sorted(items)[(offset+group_index)%len(items)])
        versions=versions[:max_snapshots]
    path=Path(output) if output else None
    if path: path.mkdir(parents=True,exist_ok=True)
    records,oracle_rows=[],[]
    oracle_versions=set(versions[i] for i in torch.linspace(0,len(versions)-1,min(oracle_tiles,len(versions))).long().tolist()) if oracle_tiles else set()
    journal=None
    if path:
        methods=('Ours','B4','B4_cost_capped','B0','B1','B2') if baselines else ('Ours','B4','B4_cost_capped')
        expected={}
        for version in versions:
            keys=set()
            for item in data.groups[version]:
                # Count contiguous tiles exactly, including captures originally
                # sampled at a different tile size.
                if 'query_tile_starts' in item:
                    indices=[i for start in item['query_tile_starts'] for i in
                             range(start,min(start+item['query_tile'],item['total_queries']))]
                    if len(indices)!=item['captured_queries']:
                        raise ValueError('Captured query metadata mismatch')
                    event=dict(query_indices=torch.tensor(indices),query_tile=item['query_tile'])
                else:
                    event=torch.load(data.root/item['event'],weights_only=True,map_location='cpu')
                for tile in range(len(query_slices(event,model.config.query_tile))):
                    keys.update((item['step_id'],tile,method,keep) for method in methods for keep in keeps)
            expected[version]=keys
        signature=identity(model,data,versions,keeps,baselines,oracle_tiles,save_masks)
        journal=EvaluationJournal(path,signature,versions,expected,oracle_versions,keeps,adopt_legacy=resume_existing)
        records,oracle_rows=journal.records,journal.oracle_rows
    raw_file=journal.raw if journal else None
    oracle_file=journal.oracle if journal else None
    begin=journal.done if journal else 0
    try:
        for snapshot_index in range(begin,len(versions)):
            version=versions[snapshot_index]
            events=data.load(version,device)
            first=events[0]
            state=model.precompute(first['k_hist'],first['v_hist'],first['token_positions'],first['head_id'],first['layer_id'])
            assignments={'Ours':state['hard']}
            if baselines:
                assignments.update({name:baseline_assignment(name,first['k_hist'],first['v_hist'],first['token_positions'],
                    model.config.target_size,model.config.max_blocks) for name in ('B0','B1','B2')})
            if path and save_masks:
                atomic_save(dict(labels=state['hard'].argmax(-1).cpu(),
                    raw_labels=state['raw_labels'].cpu() if state['raw_labels'] is not None else None,
                    diagnostics=cluster_diagnostics(state),cache_version=version),path/'partitions'/f'{version}.pt')
            for sample in events:
                for tile_index,selection in enumerate(query_slices(sample,model.config.query_tile)):
                    q=sample['q'][selection]
                    target=sample['o_dense'][selection]
                    stats={name:statistics(q,sample,a) for name,a in assignments.items()}
                    for keep in keeps:
                        route=model.route(q,state,keep)
                        costs=state['hard'].sum(0)
                        budget=route['budget']
                        masks={'Ours':route['hard_mask'],'B4':mass_selection(stats['Ours'],costs,budget)}
                        ours_kept=int((costs*route['hard_mask']).sum())
                        masks['B4_cost_capped']=mass_selection(stats['Ours'],costs,ours_kept)
                        if baselines:
                            masks.update({name:mass_selection(stats[name],assignments[name].sum(0),budget) for name in ('B0','B1','B2')})
                        for name,mask in masks.items():
                            source=name if name in assignments else 'Ours'
                            block_costs=assignments[source].sum(0)
                            kept=int((block_costs*mask).sum())
                            if kept>budget: raise AssertionError('History token budget exceeded')
                            se,energy=error_parts(stats[source].output(mask),target)
                            row={key:sample[key] for key in ('prompt_id','seed','layer_id','head_id','step_id','chunk_id','cache_version')}
                            row.update(method=name,requested_keep=keep,tile_index=tile_index,
                                query_indices=sample['query_indices'][selection].cpu().tolist(),queries=len(q),
                                history_tokens=len(sample['k_hist']),kept_tokens=kept,token_budget=budget,
                                actual_keep=kept/len(sample['k_hist']),budget_utilization=kept/max(budget,1),
                                squared_error=float(se),dense_energy=float(energy),error=float(se/(energy+1e-8)))
                            if name=='Ours':
                                soft_se,_=error_parts(stats['Ours'].output(route['soft_mask']),target)
                                row.update(soft_mask_error=float(soft_se/(energy+1e-8)),hard_soft_mask_gap=float((se-soft_se)/(energy+1e-8)))
                            if name=='B4_cost_capped':
                                row['token_difference_vs_ours']=kept-ours_kept
                            if save_masks:
                                row['selected_blocks']=torch.where(mask>0)[0].cpu().tolist()
                            records.append(row)
                            if raw_file: raw_file.write(json.dumps(row)+'\n')
                        # Sample reproducibly across snapshot versions; one tile
                        # per sampled version, covering different layers/heads.
                        if (version in oracle_versions and
                            tile_index==snapshot_index%len(query_slices(sample,model.config.query_tile)) and
                            sample['step_id']==events[snapshot_index%len(events)]['step_id']):
                            s=stats['Ours']
                            mass=masks['B4']
                            forward=forward_greedy(s,costs,budget,target)
                            refined=refine_mass(s,costs,budget,target,mass)
                            candidates=torch.stack((mass,forward,refined))
                            errors=batch_error(s,candidates,target)
                            best=candidates[int(errors.argmin())]
                            em,eo,eg=float(errors[0]),float(batch_error(s,route['hard_mask'][None],target)[0]),float(errors.min())
                            gap=em-eg
                            significant=gap>max(1e-8,.05*em)
                            row=dict(cache_version=version,prompt_id=sample['prompt_id'],seed=sample['seed'],
                                layer_id=sample['layer_id'],head_id=sample['head_id'],step_id=sample['step_id'],
                                requested_keep=keep,mass_error=em,ours_error=eo,forward_greedy_error=float(errors[1]),
                                mass_refined_error=float(errors[2]),greedy_error=eg,oracle_headroom=gap,
                                recovery=(em-eo)/gap if significant else None,significant_headroom=significant,
                                significant_rule='gap > max(1e-8,0.05*E_mass)',
                                mass_tokens=int((mass*costs).sum()),ours_tokens=ours_kept,greedy_tokens=int((best*costs).sum()),
                                budget=budget,reference='best_of_mass_forward_greedy_and_4round_mass_add_remove_swap',
                                note='Heuristic reference, not a proven optimal upper bound')
                            oracle_rows.append(row)
                            if oracle_file: oracle_file.write(json.dumps(row)+'\n')
            if journal:
                journal.commit(snapshot_index+1)
    finally:
        if journal: journal.close()
    report=dict(metrics=aggregate(records,bootstrap),snapshots=len(versions),oracle=oracle_rows,
                primary='macro_prompt_head_relative_error: normalize each layer/head, then average heads within prompt and finally prompts; global SSE/energy and tile quantiles also reported',metric_version=2)
    if path: atomic_json(path/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--split',choices=['train','val','test'],default='val')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--oracle-tiles',type=int,default=0)
    parser.add_argument('--max-snapshots',type=int)
    parser.add_argument('--resume-existing',action='store_true',help='Validate and adopt an old partial evaluation')
    args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    report=evaluate(load_router(args.checkpoint,args.device),SnapshotData(args.data,args.split),args.output,
                    oracle_tiles=args.oracle_tiles,max_snapshots=args.max_snapshots,resume_existing=args.resume_existing)
    print(json.dumps({k:v for k,v in report['metrics'].items() if k.count('/')==1},indent=2))


if __name__=='__main__': main()
