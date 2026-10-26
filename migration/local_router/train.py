"""Independent layer shards; paired K/V Attention-output-only training."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from streaming_sparse.v08.common import paired_seed,atomic_json,save,source_identity
from streaming_sparse.v08.router import InformationRouter
from streaming_sparse.v08.execution import accelerate
from migration.runtime import router_fast
from migration.paired_router.runtime import copy_pair
from .collect import event_key
from .core import score,loss


def load_data(manifest,row,layer,query_slice=None):
    cache=Path(manifest['cache']);ep=cache/'events'/event_key(row)/f'layer{layer:02d}.pt'
    e=torch.load(ep,map_location='cpu',weights_only=True,mmap=True)
    assert e['sample_id']==row['sample_id']and e['source_sha256']==manifest['source_sha256']and e['query_indices']==row['query_indices']and e['cache_format']==2
    assert e['split']==row['split']and e['prompt_id']==row['prompt_id']and e['teacher_sha256']==manifest['teacher_sha256']
    hist=torch.load(e['history_file'],map_location='cpu',weights_only=True,mmap=True)
    assert hist['layer']==layer and hist['history_tokens']==row['history_tokens']and hist['trajectory_id']==row['trajectory_id']and hist['source_sha256']==manifest['source_sha256']
    names=['q','page_logz','page_value','current_logz','current_key','current_value','exact_current_logz','exact_current_value','target']
    result={}
    for name in names:
        value=e[name]if query_slice is None else e[name][:,query_slice]
        result[name]=value.to('cuda',non_blocking=False)
    result.update(kh=hist['kh'].to('cuda'),vh=hist['vh'].to('cuda'))
    return result


def make_router(common,kind,layer):
    r=InformationRouter(kind,pairs=12).cuda();st=r.state_dict()
    for h in range(12):copy_pair(st,common,(layer-1)*12+h,h)
    st['sketch.cal_scale'].copy_(common['sketch.cal_scale']);r.load_state_dict(st)
    return r


def normalize(m,router,layer):
    rows=[r for r in m['rows']if r['split']=='train'];sums=torch.zeros(12,6,device='cuda',dtype=torch.float64);sq=torch.zeros_like(sums);n=0
    for i,pid in enumerate(m['splits']['train']):
        row=next(r for r in rows if r['prompt_id']==pid and r['generation_seed']==i%2 and r['history_tokens']==[4680,9360,14040][i%3]and r['step']==i%4)
        d=load_data(m,row,layer,slice(0,16));f=score(router,d,layer,row['step'],features_only=True).double().flatten(1,2)
        sums+=f.sum(1);sq+=f.square().sum(1);n+=f.shape[1]
    mean=sums/n;var=(sq/n-mean.square()).clamp_min(0)
    router.sketch.feature_mean.copy_(mean.float());router.sketch.feature_std.copy_(var.sqrt().clamp_min(1e-3).float())
    router.sketch.feature_count.fill_(n);router.sketch.normalization_locked.fill_(True)


@torch.no_grad()
def validate(m,routers,layer,rows,mass_only=False):
    result={k:{}for k in routers};keep=[];started=time.monotonic()
    for row in rows:
        d=load_data(m,row,layer)
        for kind,router in routers.items():
            value,info=loss(router,d,layer,row['step'],hard_only=True,mass_only=mass_only)
            result[kind].setdefault(row['prompt_id'],[]).append(float(value));keep.append(float(info['actual_keep']))
    return{k:dict(prompt_equal_nmse=sum(sum(v)/len(v)for v in byprompt.values())/len(byprompt),
        by_prompt={p:sum(v)/len(v)for p,v in byprompt.items()},actual_keep=sum(keep)/len(keep),
        paired_validation_seconds=time.monotonic()-started)for k,byprompt in result.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--benchmark',action='store_true');a=p.parse_args();root=Path(a.directory).resolve()
    m=json.loads((root/'experiment.json').read_text());assert m['source_sha256']==source_identity();assert json.loads((root/'audit.json').read_text())['passed']
    paired_seed(0);torch.cuda.set_device(0);accelerate.install();router_fast.install()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));layers=m['layers'][rank::world]
    common=torch.load(root/'common_initial.pt',map_location='cpu',weights_only=True)['router'];train=[r for r in m['rows']if r['split']=='train'];val=[r for r in m['rows']if r['split']=='val']
    output=root/'training';output.mkdir(exist_ok=True);states={};models={};optimizers={};start_time=time.monotonic()
    current={}
    def status(**kw):
        current.update(kw,rank=rank,layers=layers,updated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start_time)
        atomic_json(output/f'rank{rank}.json',current);print(json.dumps(kw),flush=True)
    if a.benchmark:
        # One cached real event, no model/counter changes persisted as training.
        row=next(r for r in m['rows']if (Path(m['cache'])/'events'/event_key(r)/'layer01.pt').exists());layer=1
        count=m['training_queries_per_event'];models={k:make_router(common,k,layer)for k in ['K','V']};d=load_data(m,row,layer,slice(0,count));times=[]
        options={k:torch.optim.AdamW(r.parameters(),lr=1e-4)for k,r in models.items()}
        for i in range(5):
            torch.cuda.synchronize();t=time.monotonic()
            for kind,router in models.items():
                router.zero_grad(set_to_none=True);value,_=loss(router,d,layer,row['step']);value.backward()
                assert torch.isfinite(value)and all(p.grad is None or torch.isfinite(p.grad).all()for p in router.parameters())
                options[kind].step()
            torch.cuda.synchronize();times.append(time.monotonic()-t)
        report=dict(passed=True,cached_event=row['sample_id'],queries=count,both_KV_single_layer_seconds=times,
            median_warm_seconds=sorted(times[1:])[2],note='No IO/backbone;8-event accumulation and3/4 layers per GPU multiply this cost; collection and validation separate.')
        atomic_json(root/'benchmark.json',report);print(json.dumps(report),flush=True);return
    cache=Path(m['cache']);receipts=list((cache/'receipts').glob('*.json'));assert len(receipts)==len(m['rows'])
    for row in m['rows']:
        receipt=json.loads((cache/'receipts'/f'{event_key(row)}.json').read_text())
        assert receipt['source_sha256']==m['source_sha256']and receipt['query_indices']==row['query_indices']and receipt['layers']==m['layers']
    status(status='preflight_passed',backbone_loaded=False,backbone_trainable_parameters=0,training_queries_per_event=m['training_queries_per_event'],
        objective=m['loss'],data_manifest_verified=True)
    # Restore complete rank state; each layer has independent optimizer/counters.
    resume=output/f'rank{rank}.recovery.pt';saved=torch.load(resume,map_location='cpu',weights_only=True)if resume.exists()else None
    if saved:assert saved['source_sha256']==m['source_sha256']
    if saved and all(st['done'] for st in saved['states'].values()):
        status(status='completed',iteration=saved['iteration'],layer_summaries=saved['states'],
            resumed=True,completed_before_resume=True)
        return
    rng=random.Random(0);pool=[];begin=0
    for layer in layers:
        status(status='initializing',active_layer=layer)
        pair={k:make_router(common,k,layer)for k in ['K','V']}
        if saved:
            for k in pair:pair[k].load_state_dict(saved['models'][layer][k])
            st=saved['states'][layer]
        else:
            with torch.no_grad():normalize(m,pair['K'],layer)
            pair['V'].load_state_dict(pair['K'].state_dict());st=dict(updates=0,done=False,curve={'K':[],'V':[]},best={'K':None,'V':None},best_step={'K':0,'V':0})
        models[layer]=pair;states[layer]=st;optimizers[layer]={k:torch.optim.AdamW(r.parameters(),lr=m['lr'],weight_decay=.01)for k,r in pair.items()}
        if saved:
            for k in pair:optimizers[layer][k].load_state_dict(saved['optimizers'][layer][k])
        else:
            for k,r in pair.items():save(output/f'layer{layer:02d}/{k}/initial.pt',dict(router=r.state_dict(),updates=0,layer=layer,information=k))
            initial=validate(m,pair,layer,val)
            for k,record in initial.items():
                st['curve'][k].append(dict(updates=0,**record));st['best'][k]=record['prompt_equal_nmse']
                save(output/f'layer{layer:02d}/{k}/best.pt',dict(router=pair[k].state_dict(),updates=0,layer=layer,information=k))
            atomic_json(output/f'layer{layer:02d}/mass_baseline.json',initial)
        status(status='initializing',active_layer=layer,initialized_layers=[*models],layer_summaries=states)
    if saved:rng.setstate(saved['rng']);pool=saved['pool'];begin=saved['iteration']
    def checkpoint(iteration):
        save(resume,dict(source_sha256=m['source_sha256'],iteration=iteration,states=states,rng=rng.getstate(),pool=pool,
            models={l:{k:r.state_dict()for k,r in pair.items()}for l,pair in models.items()},
            optimizers={l:{k:o.state_dict()for k,o in pair.items()}for l,pair in optimizers.items()}))
    checkpoint(begin);durations=[]
    for iteration in range(begin+1,m['max_updates']+1):
        if(root/'STOP').exists():checkpoint(iteration-1);status(status='paused');return
        batch=[]
        for _ in range(m['batch_events']):
            if not pool:pool=list(range(len(train)));rng.shuffle(pool)
            row=train[pool.pop()];batch.append((row,slice(0,m['training_queries_per_event'])))
        tick=time.monotonic();logs={}
        for layer in layers:
            st=states[layer]
            if st['done']:continue
            for k,optimizer in optimizers[layer].items():
                lr=m['lr']*min(1.,iteration/m['warmup'])*(.05+.95*.5*(1+math.cos(math.pi*max(0,iteration-m['warmup'])/(m['max_updates']-m['warmup']))))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
            totals={k:0. for k in ['K','V']}
            for row,sl in batch:
                d=load_data(m,row,layer,sl)
                for k,r in models[layer].items():
                    value,_=loss(r,d,layer,row['step']);assert torch.isfinite(value)
                    (value/m['batch_events']).backward();totals[k]+=float(value.detach())/m['batch_events']
            gradient_norms={}
            for k,r in models[layer].items():
                gradient_norms[k]=float(torch.nn.utils.clip_grad_norm_(r.parameters(),m['clip_per_layer'],error_if_nonfinite=True));optimizers[layer][k].step()
            st['updates']=iteration;logs[layer]=dict(loss=totals,gradient_norms=gradient_norms,lr=lr)
        torch.cuda.synchronize();seconds=time.monotonic()-tick;durations.append(seconds)
        with(output/f'rank{rank}.metrics.jsonl').open('a')as stream:stream.write(json.dumps(dict(iteration=iteration,seconds=seconds,losses=logs,sample_ids=[r['sample_id']for r,_ in batch],query_slices=[(s.start,s.stop)for _,s in batch]))+'\n')
        if iteration%m['validate_every']==0 or iteration==m['max_updates']:
            status(status='validating',iteration=iteration,layer_summaries=states)
            for layer in layers:
                st=states[layer]
                if st['done']:continue
                result=validate(m,models[layer],layer,val)
                for k,record in result.items():
                    st['curve'][k].append(dict(updates=iteration,**record))
                    if record['prompt_equal_nmse']<st['best'][k]:
                        st['best'][k]=record['prompt_equal_nmse'];st['best_step'][k]=iteration
                        save(output/f'layer{layer:02d}/{k}/best.pt',dict(router=models[layer][k].state_dict(),updates=iteration,layer=layer,information=k))
                if iteration>=m['min_updates']:
                    plateau=[]
                    for k in ['K','V']:
                        curve=st['curve'][k];old=min(r['prompt_equal_nmse']for r in curve[:-3]);recent=min(r['prompt_equal_nmse']for r in curve[-3:]);plateau.append(max(0.,old-recent)/max(old,1e-12)<.005)
                    st['done']=all(plateau)or iteration>=m['max_updates']
                atomic_json(output/f'layer{layer:02d}/validation.json',st)
                for k,r in models[layer].items():save(output/f'layer{layer:02d}/{k}/last.pt',dict(router=r.state_dict(),updates=iteration,layer=layer,information=k))
        checkpoint(iteration)
        status(status='training',iteration=iteration,seconds_last=seconds,seconds_mean=sum(durations[-20:])/len(durations[-20:]),
            layer_summaries=states,sample_ids=[r['sample_id']for r,_ in batch])
        if all(s['done']for s in states.values()):break
    status(status='completed',iteration=iteration,layer_summaries=states)


if __name__=='__main__':main()
