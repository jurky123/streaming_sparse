import argparse,json,math,random,time,traceback
from pathlib import Path
import torch
from .setup import ROOT,VARIANTS
from .data import Events
from .model import QueryRouter
from ..common import save,atomic_json,update,paired_seed
from ..opt12.precision import clip_grad_norm
from ..opt18.train import append

def summary(r):return {k:v for k,v in r.items()if k!='rows'}
@torch.no_grad()
def evaluate(model,events,items=None):
 rows=[]
 for item in(events.val if items is None else items):
  q,raw,truth,targets=events.gpu(item);prev=events.previous(item,model.options['refresh']);out,d=model.deployed(q,raw,targets,prev);sse=(out-truth).square().sum();energy=truth.square().sum().clamp_min(1e-20)
  assert torch.isfinite(out).all();rows.append(dict(sample_id=item['row']['sample_id'],split=item['row']['split'],nmse=float(sse/energy),sse=float(sse),energy=float(energy),keep=float(d['keep']),executed=float(d['executed'])))
 return dict(nmse=sum(x['nmse']for x in rows)/len(rows),energy_weighted_nmse=sum(x['sse']for x in rows)/sum(x['energy']for x in rows),keep=sum(x['keep']for x in rows)/len(rows),executed=sum(x['executed']for x in rows)/len(rows),worst_event_nmse=max(x['nmse']for x in rows),rows=rows)
def checkpoint(model,step,best):return dict(model=model.state_dict(),layer=model.layer,variant=model.variant,updates=step,best_value=best,best_metric='actual_native_attention_nmse',options=model.options,backward='FP32 page-gate STE; discrete coarse candidates held fixed; real native output in loss; bounded local pooling gradients; behavior trees fixed during continuous adaptation',backbone_frozen=True)
def run(layer,name,total,minimum):
 m=json.loads((ROOT/'experiment.json').read_text());folder=ROOT/f'training/layer{layer:02d}/{name}';folder.mkdir(parents=True,exist_ok=True);status=folder/'status.json'
 paired_seed(30000+layer);rng=random.Random(30000+layer);update(status,status='loading',updates=0,total=total,layer=layer,variant=name)
 events=Events(m,layer);model=QueryRouter(m,layer,name).cuda();params=[dict(params=model.routing.parameters(),lr=3e-5,base_lr=3e-5)]
 if model.options['mode']=='local':params.append(dict(params=[model.query_weight],lr=1e-4,base_lr=1e-4))
 opt=torch.optim.AdamW(params,weight_decay=0.);recovery=folder/'recovery.pt';start=0;stale=0
 if recovery.exists():
  rec=torch.load(recovery,map_location='cpu',weights_only=True);model.load_state_dict(rec['model']);opt.load_state_dict(rec['optimizer']);start=rec['updates'];best=rec['best'];initial=rec['initial'];significant=rec['significant'];stale=rec['stale'];curves=rec['curves'];rng.setstate(rec['rng'])
  if total>rec.get('total',total):stale=0
 else:
  model.initialize_groups()
  update(status,status='initial_validation',updates=0,total=total,layer=layer,variant=name);result=evaluate(model,events);best=initial=significant=result['nmse'];curves=[dict(updates=0,**summary(result))]
  for label in ['initial','best']:
   save(folder/(label+'.pt'),checkpoint(model,0,best));atomic_json(folder/(label+'_validation.json'),dict(updates=0,**result))
  atomic_json(folder/'curve.json',curves);print('initial',layer,name,summary(result),flush=True)
 def recover(step):save(recovery,dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=step,total=total,best=best,initial=initial,significant=significant,stale=stale,curves=curves,rng=rng.getstate()))
 append(folder/'resume_history.jsonl',dict(start_updates=start,target=total,minimum=minimum));last=start;reason='max_updates'
 for step in range(start+1,total+1):
  if(ROOT/'STOP').exists():reason='stopped';break
  t0=time.perf_counter();opt.zero_grad(set_to_none=True);ratio=min(1.,step/50)*(.2+.8*.5*(1+math.cos(math.pi*step/6000)))
  for group in opt.param_groups:group['lr']=group['base_lr']*ratio
  chosen=rng.sample(events.train,2);metrics=dict(actual_nmse=0.,keep=0.,executed=0.)
  for item in chosen:
   q,raw,truth,targets=events.gpu(item);prev=events.previous(item,model.options['refresh']);out,d=model(q,raw,targets,prev);loss=(out-truth).square().sum()/truth.square().sum().clamp_min(1e-20)
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite actual attention loss')
   (loss/2).backward();metrics['actual_nmse']+=float(loss.detach())/2
   for key in ['keep','executed']:metrics[key]+=float(d[key])/2
   del q,raw,truth,prev,out,d,loss
  norm=clip_grad_norm(model.trainable(),1.);metrics['query_gradient']=float(model.query_weight.grad.norm())if model.query_weight.grad is not None else 0.;opt.step();torch.cuda.synchronize();last=step
  metrics.update(gradient_norm=float(norm),update_seconds=time.perf_counter()-t0,learning_rate=opt.param_groups[0]['lr']);append(folder/'train.jsonl',dict(updates=step,**metrics,event_ids=[x['row']['sample_id']for x in chosen]))
  update(status,status='training',updates=step,total=total,layer=layer,variant=name,best=best,initial=initial,losses=metrics,peak_gb=torch.cuda.max_memory_allocated()/1e9)
  if step==start+1 or step%25==0:print('train',step,metrics,flush=True)
  # Behavior-distilled trees are frozen during this continuous adaptation phase.
  if step%100==0 or step==total:
   update(status,status='validating',updates=step,total=total,layer=layer,variant=name,best=best,initial=initial);result=evaluate(model,events);assert result['executed']<.22
   curves.append(dict(updates=step,**summary(result)));atomic_json(folder/'curve.json',curves);atomic_json(folder/f'validation/update{step:06d}.json',dict(updates=step,**result))
   if result['nmse']<best:
    best=result['nmse'];save(folder/'best.pt',checkpoint(model,step,best));atomic_json(folder/'best_validation.json',dict(updates=step,**result))
   if result['nmse']<significant*.999:significant=result['nmse'];stale=0
   else:stale+=1
   print('validation',step,summary(result),'best',best,flush=True)
   from .plot import plot
   plot(folder)
  plateau=step>=minimum and stale>=8
  if step%50==0 or plateau or step==total:recover(step)
  if plateau:reason='validation_plateau';break
 recover(last);save(folder/'last.pt',checkpoint(model,last,best));record=dict(layer=layer,variant=name,updates=last,requested=total,reason=reason,initial=initial,best=best,best_metric='actual_native_attention_nmse',convergence='plateau'if reason=='validation_plateau'else'not_asserted')
 atomic_json(folder/f'completed_{total}.json',record);atomic_json(folder/'completed.json',record);update(status,status='stopped'if reason=='stopped'else'completed',total=total,**record)
 if reason=='stopped':raise SystemExit(2)
def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--variant',required=True);p.add_argument('--updates',type=int,default=2000);p.add_argument('--minimum',type=int,default=1000);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert json.loads((ROOT/'preflight.json').read_text())['passed']
 try:run(a.layer,a.variant,a.updates,a.minimum)
 except Exception as e:
  folder=ROOT/f'training/layer{a.layer:02d}/{a.variant}';update(folder/'status.json',status='failed',layer=a.layer,variant=a.variant,error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
