import argparse,json,math,os,random,time,traceback
from pathlib import Path
import torch
from .setup import ROOT
from .model import TileRouter
from ..common import config,save,atomic_json,update,paired_seed,sha
from ..opt17.data import Events
from ..opt12.precision import clip_grad_norm
from ..opt18.train import evaluate,summary,append

def checkpoint(model,step,best):
    return dict(model=model.state_dict(),parameters={k:v.detach() for k,v in model.dictionary().items()},
        layer=model.layer,variant=model.variant,updates=step,best_value=best,best_metric='actual_deployment_attention_nmse',
        options=model.options,tile_budget=model.budget,tile_queries=32,full_q_context=4680,
        backward='biased tile-probability/coarse-gate STE; real attention-output loss; fixed grouping trees',
        backward_version='stable_log_softmax_v2')

def run(m,layer,variant,max_updates=None):
    folder=ROOT/f'training/layer{layer:02d}/{variant}';folder.mkdir(parents=True,exist_ok=True)
    status=folder/'status.json';total=max_updates or m['max_updates'];paired_seed(20000+layer);rng=random.Random(20000+layer)
    update(status,status='loading',updates=0,total=total,layer=layer,variant=variant)
    events=Events(m,layer);model=TileRouter(m,layer,variant).cuda()
    initial_source=ROOT/f'initialization/layer{layer:02d}.pt'
    model.load_state_dict(torch.load(initial_source,map_location='cuda',weights_only=True)['model'])
    atomic_json(folder/'initial_source.json',dict(path=str(initial_source),sha256=sha(initial_source),shared_initialization=True))
    opt=torch.optim.AdamW(model.routing.parameters(),lr=m['learning_rate_router'],weight_decay=0.)
    recovery=folder/'recovery.pt';start=0;curves=[];stale=0
    if recovery.exists():
        rec=torch.load(recovery,map_location='cpu',weights_only=True);model.load_state_dict(rec['model']);opt.load_state_dict(rec['optimizer'])
        start=rec['updates'];best=rec['best'];initial=rec['initial'];significant=rec['significant'];stale=rec['stale'];curves=rec['curves'];rng.setstate(rec['rng'])
    else:
        update(status,status='initial_validation',updates=0,total=total,layer=layer,variant=variant)
        result=evaluate(model,events);best=initial=significant=result['nmse'];curves=[dict(updates=0,**summary(result))]
        assert result['executed']<=model.budget+.012
        for name in ['initial','best']:
            save(folder/(name+'.pt'),checkpoint(model,0,best));atomic_json(folder/(name+'_validation.json'),dict(updates=0,**result))
        atomic_json(folder/'curve.json',curves);print('initial',layer,variant,summary(result),flush=True)
    def recover(step):
        save(recovery,dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=step,best=best,initial=initial,
            significant=significant,stale=stale,curves=curves,rng=rng.getstate()))
    append(folder/'resume_history.jsonl',dict(start_updates=start,max_updates=total,backward_version='stable_log_softmax_v2'))
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
            if not torch.isfinite(loss):raise FloatingPointError((step,item['row']['sample_id'],'nonfinite loss'))
            (loss/len(chosen)).backward();metrics['actual_nmse']+=float(loss.detach())/len(chosen)
            for k in ['keep','executed']:metrics[k]+=float(d[k])/len(chosen)
            del q,raw,truth,out,d,loss
        norm=clip_grad_norm(model.routing.parameters(),1.);opt.step();torch.cuda.synchronize();last=step
        metrics.update(gradient_norm=float(norm),learning_rate=opt.param_groups[0]['lr'],update_seconds=time.perf_counter()-t0)
        append(folder/'train.jsonl',dict(updates=step,**metrics,event_ids=[x['row']['sample_id'] for x in chosen]))
        update(status,status='training',updates=step,total=total,layer=layer,variant=variant,best=best,initial=initial,
               losses=metrics,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        if step==start+1 or step%25==0:print('train',layer,variant,step,metrics,flush=True)
        if step%m['validate_every']==0 or step==total:
            update(status,status='validating',updates=step,total=total,layer=layer,variant=variant,best=best,initial=initial)
            result=evaluate(model,events);eligible=result['executed']<=model.budget+.012
            curves.append(dict(updates=step,eligible=eligible,**summary(result)));atomic_json(folder/'curve.json',curves)
            atomic_json(folder/f'validation/update{step:06d}.json',dict(updates=step,eligible=eligible,**result))
            if eligible and result['nmse']<best:
                best=result['nmse'];save(folder/'best.pt',checkpoint(model,step,best));atomic_json(folder/'best_validation.json',dict(updates=step,**result))
            if eligible and result['nmse']<significant*(1-m['min_relative_improvement']):significant=result['nmse'];stale=0
            else:stale+=1
            from ..opt17.plot import plot
            plot(folder);print('validation',layer,variant,step,summary(result),'best',best,flush=True)
        plateau=step>=m['min_updates'] and stale>=m['patience_validations']
        if step%m['checkpoint_every']==0 or plateau or step==total:recover(step)
        if plateau:reason='validation_plateau';break
    recover(last);save(folder/'last.pt',checkpoint(model,last,best))
    if reason=='stop_file':update(status,status='stopped',updates=last,total=total,layer=layer,variant=variant);return
    atomic_json(folder/'completed.json',dict(layer=layer,variant=variant,updates=last,reason=reason,initial=initial,best=best,
        tile_budget=model.budget,best_metric='actual_deployment_attention_nmse'))
    update(status,status='completed',updates=last,total=total,layer=layer,variant=variant,best=best,initial=initial)

def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--variant',required=True);p.add_argument('--max-updates',type=int);a=p.parse_args()
    _,m=config(ROOT);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    try:run(m,a.layer,a.variant,a.max_updates)
    except Exception as e:
        path=ROOT/f'training/layer{a.layer:02d}/{a.variant}/status.json';old=json.loads(path.read_text()) if path.exists() else {};old.pop('updated_at',None)
        update(path,**dict(old,status='failed',error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
