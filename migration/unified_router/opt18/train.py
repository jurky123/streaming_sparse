import argparse,json,math,os,random,time,traceback
from pathlib import Path
import torch
from .setup import ROOT
from ..common import config,save,atomic_json,update,paired_seed
from ..opt12.precision import clip_grad_norm
from ..opt17.data import Events
from ..opt17.model import JointGroupedRouter
from ..opt16.grouping import Tree

def append(path,record):
    with path.open('a')as f:f.write(json.dumps(dict(time=time.time(),**record),allow_nan=False)+'\n');f.flush()

@torch.no_grad()
def evaluate(model,events,items=None,fraction=None,representatives=None):
    rows=[]
    for item in (events.val if items is None else items):
        q,raw,truth,targets=events.gpu(item)
        y,d=model.deployed(q,raw,targets,fraction,representatives)
        sse=(y-truth).square().sum();energy=truth.square().sum().clamp_min(1e-20)
        rows.append(dict(sample_id=item['row']['sample_id'],split=item['row']['split'],nmse=float(sse/energy),sse=float(sse),energy=float(energy),keep=float(d['keep']),executed=float(d['executed'])))
    return dict(nmse=sum(r['nmse']for r in rows)/len(rows),energy_weighted_nmse=sum(r['sse']for r in rows)/sum(r['energy']for r in rows),
        keep=sum(r['keep']for r in rows)/len(rows),executed=sum(r['executed']for r in rows)/len(rows),worst_event_nmse=max(r['nmse']for r in rows),rows=rows)

def summary(result):return {k:v for k,v in result.items()if k!='rows'}

def checkpoint(model,step,best):
    return dict(model=model.state_dict(),parameters={k:v.detach()for k,v in model.dictionary().items()},layer=model.layer,variant=model.variant,
        updates=step,best_value=best,best_metric='actual_deployment_attention_nmse',options=model.options,
        backward='FP32 gate STE; original128d page attention Jacobian; best grouping trees frozen',full_q_context=4680)

@torch.no_grad()
def refresh(model,events,m,step,folder):
    seed=17000+model.layer+step
    q,k,cal_ids=events.calibration_tensors(seed,m['calibration_events'])
    pool=[item for item in events.train if item['row']['sample_id']not in cal_ids]
    probe=random.Random(seed).sample(pool,m['tree_probe_events'])
    before=evaluate(model,events,probe)
    old=[Tree(model.tree(s).weight.clone(),model.tree(s).threshold.clone(),model.depth)for s in ['q','k']]
    proposed=model.propose_trees(q,k,projected=True);model.set_trees(*proposed)
    after=evaluate(model,events,probe)
    accepted=after['nmse']<before['nmse']*(1-m['tree_min_relative_improvement']) and after['executed']<=before['executed']*m['tree_max_relative_tile_increase']
    if not accepted:model.set_trees(*old)
    record=dict(updates=step,accepted=accepted,split='train',calibration_events=cal_ids,probe_events=[x['row']['sample_id']for x in probe],before=summary(before),after=summary(after))
    append(folder/'tree_updates.jsonl',record)
    return record

def run(m,layer,variant,rank,max_updates=None):
    folder=ROOT/f'training/layer{layer:02d}/{variant}';folder.mkdir(parents=True,exist_ok=True)
    status=ROOT/f'training/rank{rank}.json';total=max_updates or m['max_updates']
    paired_seed(17000+layer);rng=random.Random(17000+layer)
    update(status,status='loading',layer=layer,variant=variant,updates=0,total=total)
    events=Events(m,layer);model=JointGroupedRouter(m,layer,variant).cuda()
    source=Path(m['parent_opt17'])/f'training/layer{layer:02d}/{variant}/best.pt'
    parent=torch.load(source,map_location='cuda',weights_only=True)
    assert parent['best_metric']=='actual_deployment_attention_nmse'
    model.load_state_dict(parent['model'])
    atomic_json(folder/'initial_source.json',dict(path=str(source),parent_updates=parent['updates'],parent_best=parent['best_value'],optimizer_reset=True,trees_frozen=True))
    opt=torch.optim.AdamW(model.routing.parameters(),lr=m['learning_rate_router'],weight_decay=0.)
    recovery=folder/'recovery.pt';start=0;curves=[];stale=0
    if recovery.exists():
        rec=torch.load(recovery,map_location='cpu',weights_only=True);model.load_state_dict(rec['model']);opt.load_state_dict(rec['optimizer']);start=rec['updates']
        best=rec['best'];initial=rec['initial'];significant=rec['significant'];stale=rec['stale'];curves=rec['curves'];rng.setstate(rec['rng'])
    else:
        update(status,status='initial_validation',layer=layer,variant=variant,updates=0,total=total)
        result=evaluate(model,events)
        assert abs(result['nmse']-parent['best_value'])<1e-7,(layer,variant,'initial mismatch',result['nmse'],parent['best_value'])
        best=initial=significant=result['nmse'];curves=[dict(updates=0,**summary(result))]
        for name in ['initial','best']:
            save(folder/(name+'.pt'),checkpoint(model,0,best));atomic_json(folder/(name+'_validation.json'),dict(updates=0,**result))
        atomic_json(folder/'curve.json',curves)
        print('initial',layer,variant,summary(result),flush=True)
    def recover(step):save(recovery,dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=step,best=best,initial=initial,significant=significant,stale=stale,curves=curves,rng=rng.getstate()))
    append(folder/'resume_history.jsonl',dict(start_updates=start,rank=rank,full_q_context=True))
    last=start;reason='max_updates'
    for step in range(start+1,total+1):
        if (ROOT/'STOP').exists():reason='stop_file';break
        t0=time.perf_counter();opt.zero_grad(set_to_none=True)
        ratio=min(1.,step/m['warmup_updates'])*(.2+.8*.5*(1+math.cos(math.pi*step/m['max_updates'])))
        opt.param_groups[0]['lr']=m['learning_rate_router']*ratio
        chosen=rng.sample(events.train,m['batch_events']);metrics=dict(actual_nmse=0.,keep=0.,executed=0.)
        for item in chosen:
            q,raw,truth,targets=events.gpu(item)
            out,d=model(q,raw,targets,m['gate_temperature']);loss=(out-truth).square().sum()/truth.square().sum().clamp_min(1e-20)
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at {step}: {item["row"]["sample_id"]}')
            (loss/len(chosen)).backward()
            metrics['actual_nmse']+=float(loss.detach())/len(chosen)
            for key in ['keep','executed']:metrics[key]+=float(d[key])/len(chosen)
            del out,d,loss,q,raw,truth
        norm=clip_grad_norm(model.routing.parameters(),1.)
        opt.step();torch.cuda.synchronize();last=step
        metrics.update(gradient_norm=float(norm),learning_rate=opt.param_groups[0]['lr'],update_seconds=time.perf_counter()-t0)
        append(folder/'train.jsonl',dict(updates=step,**metrics,event_ids=[x['row']['sample_id']for x in chosen]))
        update(status,status='training',layer=layer,variant=variant,updates=step,total=total,losses=metrics,best=best,initial=initial,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        if step==start+1 or step%25==0:print('train',layer,variant,step,metrics,flush=True)
        if m['tree_refresh_every'] and step%m['tree_refresh_every']==0:
            update(status,status='tree_refresh',layer=layer,variant=variant,updates=step,total=total,best=best,initial=initial)
            print('tree',refresh(model,events,m,step,folder),flush=True)
        if step%m['validate_every']==0 or step==total:
            update(status,status='validating',layer=layer,variant=variant,updates=step,total=total,best=best,initial=initial)
            result=evaluate(model,events);eligible=result['keep']<=.16
            curves.append(dict(updates=step,eligible=eligible,**summary(result)));atomic_json(folder/'curve.json',curves)
            atomic_json(folder/f'validation/update{step:06d}.json',dict(updates=step,eligible=eligible,**result))
            if eligible and result['nmse']<best:
                best=result['nmse'];save(folder/'best.pt',checkpoint(model,step,best));atomic_json(folder/'best_validation.json',dict(updates=step,**result))
            if eligible and result['nmse']<significant*(1-m['min_relative_improvement']):significant=result['nmse'];stale=0
            else:stale+=1
            from ..opt17.plot import plot
            plot(folder)
            print('validation',layer,variant,step,summary(result),'best',best,flush=True)
        plateau=step>=m['min_updates'] and stale>=m['patience_validations']
        if step%m['checkpoint_every']==0 or plateau or step==total:recover(step)
        if plateau:reason='validation_plateau';break
    recover(last);save(folder/'last.pt',checkpoint(model,last,best))
    if reason=='stop_file':update(status,status='stopped',layer=layer,variant=variant,updates=last,total=total);return
    # Only after training, evaluate the chosen checkpoint across wider budgets.
    best_state=torch.load(folder/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(best_state['model'])
    budgets=[]
    update(status,status='budget_validation',layer=layer,variant=variant,updates=last,total=total,best=best,initial=initial)
    for frac in m['validation_extra_candidate_fractions']:
        result=evaluate(model,events,fraction=frac,representatives=16)
        atomic_json(folder/f'budget_validation/candidate{int(frac*100)}_r16.json',result);budgets.append(dict(candidate_fraction=frac,representatives=16,**summary(result)))
    atomic_json(folder/'completed.json',dict(layer=layer,variant=variant,updates=last,reason=reason,initial=initial,best=best,budgets=budgets))
    update(status,status='completed',layer=layer,variant=variant,updates=last,total=total,best=best,initial=initial)

def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--variant',required=True);p.add_argument('--max-updates',type=int);a=p.parse_args()
    _,m=config(ROOT);rank=int(os.environ.get('RANK',0));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.cache_size_limit=128;torch._dynamo.config.accumulated_cache_size_limit=1024
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    try:run(m,a.layer,a.variant,rank,a.max_updates)
    except Exception as e:
        f=ROOT/f'training/rank{rank}.json';old=json.loads(f.read_text())if f.exists()else{};old.pop('updated_at',None)
        update(f,**dict(old,status='failed',error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
