"""Frozen evaluation shards; disjoint events, original single-event arithmetic."""
import argparse,collections,json,os,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,OUT,E24,PRE,NAMES,case_files,atomic,digest
from .train import Bundle
from .training_execution import load_cpu,to_device,Prefetch
from ..review_v1.ops import Summary
from ..e24_joint_representatives.core import summary,contributions,route,gains,objective,predicted
from ..learned_cache.io import RunLock
CACHE=OUT/'execution_optimization/old_validation'

def baseline_source():
    return dict(dataset=digest(OUT/'dataset_audit.json'),summary_checkpoint=digest(PRE),reference_code=digest(ROOT/'local_experiments/e24_joint_representatives/core.py'))

@torch.no_grad()
def old_value(bundle,d):
    m=bundle.layers[d['key']];c=contributions(d['q'],d['old'],m.basis,d['log_scale']);ix=torch.linspace(0,63,16,device='cuda').round().long()
    mask,_,_=route(d['q'][ix],d['old'],m,d['own'])
    return float(objective(c,d['N'],d['Z'],d['dense'],mask))

@torch.no_grad()
def event(bundle,d,path,old):
    qs=torch.linspace(0,63,16,device='cuda').round().long();m=bundle.layers[d['key']]
    s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale'])
    mask,r,a=route(d['q'][qs],s,m,d['own']);cp={n:v[qs] for n,v in c.items()};z,u,beta=bundle.reader(d['q'][qs],s,cp)
    from .stable_targets import proxy_gains,true_gains,safe_objective
    score,_=proxy_gains(bundle.reader,d['q'][qs],s,cp,mask,r,a,z,u,beta);j=int(score.argmax());gain=true_gains(c,d,s,m,mask,r[j:j+1],a[j:j+1])[0]
    return dict(event=str(path.relative_to(OUT)),key=d['key'],trajectory=d['ev']['trajectory'],old=old,initial=float(safe_objective(c,d,s,m,mask)),executed_gain=float(gain) if score[j]>0 else 0.)

@torch.no_grad()
def worker(mode,shard,shards,output,checkpoint=None,seed=17):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    output=Path(output);lock=RunLock(output.parent/f'part{shard}_lock');begin=time.perf_counter();prefetch=None
    try:
        source=baseline_source() if mode=='baseline' else dict(weights=digest(checkpoint),baseline=digest(CACHE/'baseline.json'))
        if output.exists():
            assert json.loads(output.read_text())['source']==source;return
        bundle=Bundle(seed)
        if checkpoint:bundle.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model'])
        bundle.requires_grad_(False);paths=case_files('development');assigned=list(range(shard,len(paths),shards));rows=[]
        old=json.loads((CACHE/'baseline.json').read_text())['values'] if mode!='baseline' else None
        prefetch=Prefetch(paths,assigned,workers=2,depth=8)
        for i,index in enumerate(assigned):
            d=prefetch.get(i);path=paths[index]
            if mode=='baseline':
                cpu=load_cpu(path);d['old']=Summary(*(cpu[2]['summary'][n].cuda() for n in NAMES))
                row=dict(event=str(path.relative_to(OUT)),key=d['key'],old=old_value(bundle,d))
            else:row=event(bundle,d,path,old[str(path.relative_to(OUT))])
            rows.append(row)
            if (i+1)%96==0:
                atomic(output.with_suffix('.status.json'),dict(mode=mode,done=i+1,total=len(assigned),seconds=time.perf_counter()-begin,pid=os.getpid()));print(mode,shard,i+1,round(time.perf_counter()-begin),flush=True)
        atomic(output,dict(source=source,rows=rows,seconds=time.perf_counter()-begin,pid=os.getpid(),mode=mode))
    finally:
        if prefetch is not None:prefetch.close()
        lock.close()

def merge_baseline(shards=6):
    values={};keys=collections.Counter();source=baseline_source()
    for shard in range(shards):
        rec=json.loads((CACHE/f'part{shard}.json').read_text());assert rec['source']==source
        for row in rec['rows']:
            assert row['event'] not in values;values[row['event']]=row['old'];keys[row['key']]+=1
    assert len(values)==8352 and len(keys)==348 and set(keys.values())=={24}
    atomic(CACHE/'baseline.json',dict(source=source,values=values,coverage=dict(keys),seed_independent=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','checkpoint','merge'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);p.add_argument('--output');p.add_argument('--checkpoint');p.add_argument('--seed',type=int,default=17);a=p.parse_args()
    if a.mode=='merge':merge_baseline(a.shards)
    else:worker(a.mode,a.shard,a.shards,a.output,a.checkpoint,a.seed)
