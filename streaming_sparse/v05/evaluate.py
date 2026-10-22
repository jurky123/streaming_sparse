"""Paired per-query evaluation; signed interventions never provide training targets."""
import argparse
import json
import math
from pathlib import Path
import torch
from .common import SnapshotData, DEFAULT_DATA, source_identity, file_sha, atomic_json, protect_output
from .primitives import statistics, select, mass_mask
from .model import load_router
from .metrics import summarize, error, trace_summary, distribution
from ..evaluation.baselines import spatial_assignment
from ..capture.dataset_io import query_slices
from ..v03.metrics import partition_diagnostics


class FrozenCache:
    """Phase1 cache: immutable H once per snapshot; statistics cached on CPU per tile."""
    def __init__(self,model,data,device):
        self.model,self.data,self.device=model,data,device
        self.states={};self.events={};self.stats={};self.calls=0

    @torch.no_grad()
    def get(self,version):
        if version not in self.events:
            events=self.data.load(version,'cpu')
            self.events[version]=events
            first=events[0]
            state=self.model.precompute(first['k_hist'].to(self.device),first['v_hist'].to(self.device),
                    head_id=first['head_id'],layer_id=first['layer_id'])
            self.calls+=1
            self.states[version]={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in state.items()
                                  if k not in ('features','probabilities','raw_labels','usage_diagnostic')}
            self.states[version]['partition_diagnostic']=partition_diagnostics(state)
        state={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in self.states[version].items()}
        events=[{k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in event.items()} for event in self.events[version]]
        return events,state

    @torch.no_grad()
    def stat(self,version,event,tile,selection,state):
        key=(version,event['step_id'],tile)
        if key not in self.stats:
            self.stats[key]=statistics(event['q'][selection],event,state['assignment']).to('cpu')
        return self.stats[key].to(self.device)


@torch.no_grad()
def evaluate(model,data,selected,output,keeps=(.2,),cache=None,diagnostics=False,checkpoint_sha=None):
    output=protect_output(output);output.mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device
    identity=dict(source_sha256=source_identity(),data_fingerprint=data.fingerprint(),split=data.split,
                  snapshots=sorted(selected),keeps=list(keeps),arms=list(model.arms),
                  diagnostics=diagnostics,checkpoint_sha256=checkpoint_sha)
    meta=output/'identity.json'
    if meta.exists() and json.loads(meta.read_text())!=identity:raise ValueError('Evaluation identity mismatch')
    atomic_json(meta,identity)
    committed=output/'snapshot_commits';committed.mkdir(exist_ok=True)
    model.eval();partitions=[];diagnostic_summaries=[]
    for index,version in enumerate(sorted(selected)):
        done=committed/f'{version}.json'
        if done.exists():
            saved=json.loads(done.read_text())
            if saved.get('identity')!=identity['source_sha256'] or len(saved['rows'])!=32*len(keeps):raise ValueError('Corrupt snapshot commit')
        else:
            if cache:events,state=cache.get(version)
            else:
                events=data.load(version,device);first=events[0]
                state=model.precompute(first['k_hist'],first['v_hist'],head_id=first['head_id'],layer_id=first['layer_id'])
            costs=state['hard'].sum(0)
            partition=state.get('partition_diagnostic') or partition_diagnostics(state)
            spatial=spatial_assignment(events[0]['token_positions'],64) if diagnostics else None
            rows=[];ds=[]
            for event in events:
                for tile,selection in enumerate(query_slices(event,16)):
                    q=event['q'][selection];target=event['o_dense'][selection]
                    stat=cache.stat(version,event,tile,selection,state) if cache else statistics(q,event,state['assignment'])
                    predictions=model.predict(q,state,trace=diagnostics)
                    if diagnostics:
                        trace={name:trace_summary(route,state) for name,route in predictions.items()}
                        ds.append(trace)
                        spatial_stat=statistics(q,event,spatial)
                        interventions={mode:model.predict(q,state,intervention=mode,only=('R3',))['R3']
                                       for mode in ('positive','absolute')}
                    for keep in keeps:
                        routes=model.route(q,state,keep,predictions=predictions)
                        budget=math.floor(keep*len(state['hard']))
                        masks={name:route['hard_mask'] for name,route in routes.items()}
                        masks['Mass']=mass_mask(stat,costs,budget,state['block_ids'])
                        masks.update({name+'_soft':route['soft_mask'] for name,route in routes.items()})
                        if diagnostics:
                            masks.update({name+'_relevance':select(route['relevance'],costs,budget,state['block_ids'])
                                          for name,route in routes.items() if name!='R1'})
                            masks.update({'R3_'+mode:select(value['scores'],costs,budget,state['block_ids']) for mode,value in interventions.items()})
                        errors={name:error(stat.output(mask),target) for name,mask in masks.items()}
                        retained={name:(mask*costs).sum(-1).long().cpu().tolist() for name,mask in masks.items() if not name.endswith('_soft')}
                        if diagnostics:
                            sm=mass_mask(spatial_stat,spatial.sum(0),budget)
                            errors['SpatialMass']=error(spatial_stat.output(sm),target)
                            retained['SpatialMass']=(sm*spatial.sum(0)).sum(-1).long().cpu().tolist()
                        row={k:event[k] for k in ('prompt_id','seed','layer_id','head_id','step_id')}
                        row.update(cache_version=version,tile_id=tile,keep=keep,history_tokens=len(state['hard']),
                                   budget=budget,partition_hash=state['partition_hash'],block_count=len(costs),
                                   errors=errors,retained=retained)
                        if diagnostics:
                            for name in ('R2','R3'):
                                row[f'correction_changed_mask_fraction_{name}']=float((masks[name]!=masks[name+'_relevance']).any(-1).float().mean())
                            for mode in ('positive','absolute'):
                                row[f'{mode}_intervention_changed_mask_fraction']=float((masks['R3']!=masks['R3_'+mode]).any(-1).float().mean())
                        rows.append(row)
            saved=dict(identity=identity['source_sha256'],rows=rows,partition=partition,diagnostics=ds)
            atomic_json(done,saved)
        partitions.append(saved['partition']);diagnostic_summaries.extend(saved['diagnostics'])
        atomic_json(output/'progress.json',dict(completed=index+1,total=len(selected),split=data.split))
    all_rows=[]
    for version in sorted(selected):all_rows.extend(json.loads((committed/f'{version}.json').read_text())['rows'])
    summary=dict(identity=identity,metrics=summarize(all_rows),capacity=model.capacity(),
                 partitions={key:distribution([p[key] for p in partitions]) for key in
                             ('blocks','repair_fraction','assignment_entropy','block_min','block_max')},
                 query_masks='Independent per raw query; tile metrics aggregate the same16 captured queries as v0.4',
                 phase1_cache_precompute_calls=cache.calls if cache else None)
    if diagnostics:
        scalars={}
        def flatten(obj,prefix=''):
            if isinstance(obj,dict):
                for k,v in obj.items():yield from flatten(v,prefix+'/'+k)
            elif isinstance(obj,list):
                for i,v in enumerate(obj):yield from flatten(v,prefix+'/'+str(i))
            elif isinstance(obj,(int,float)):yield prefix,float(obj)
        for item in diagnostic_summaries:
            for key,value in flatten(item):scalars.setdefault(key,[]).append(value)
        summary['representation_diagnostics']={k:distribution(v) for k,v in scalars.items()}
    atomic_json(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--data',default=str(DEFAULT_DATA))
    parser.add_argument('--split',choices=('train','val','test'),required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one GPU')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model,value=load_router(args.checkpoint,'cuda')
    if value['identity']['source_sha256']!=source_identity() or value['phase']!='phase2':raise ValueError('Wrong checkpoint source/phase')
    data=SnapshotData(args.data,args.split)
    summary=evaluate(model,data,sorted(data.groups),args.output,keeps=(.3,.2,.1),diagnostics=True,checkpoint_sha=file_sha(args.checkpoint))
    print(json.dumps({k:v for k,v in summary['metrics'].items() if '/' not in k}),flush=True)


if __name__=='__main__':main()
