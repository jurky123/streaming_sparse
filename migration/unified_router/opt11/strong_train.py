"""Warm-start head fine-tuning; select best only by actual deployed hard AV error."""
import argparse,copy,gc,json,math,os,random,time,traceback
from pathlib import Path
import torch
from ..common import config,load_raw,save,atomic_json,update,paired_seed,sha
from .data import Builder,initial_head,load_group,attend,dense,labels,feature_path,device,cpu,VARIANTS
from .head import AlignedHead,exact,apply,subset

@torch.no_grad()
def make_features(root,m,layer,status):
 builder=Builder(m['parent_experiment'],layer,m['monitor_beta']);head=initial_head(m['parent_experiment'],layer)
 for i,row in enumerate(m['rows']):
  if (root/'STOP').exists():raise RuntimeError('STOP requested')
  path=feature_path(root,row,layer)
  if not path.exists():
   q=load_group(root,row,layer);raw=load_raw(m['parent_experiment'],row,layer);d,_=builder.inputs(q,raw,row['step']);truth=dense(q,raw)
   if row['split']=='train':
    losses,energy,base=labels(head,d,q,raw,truth,row['target_query_positions']);inputs=subset(d,row['target_query_positions'])
    payload=dict(inputs=cpu(inputs),labels=cpu(losses),energy=energy.cpu(),sample_id=row['sample_id'],split='train',target_query_positions=row['target_query_positions'])
   else:payload=dict(inputs=cpu(d),q=q.cpu(),truth=truth.cpu(),sample_id=row['sample_id'],split='val')
   save(path,payload);del d,q,raw,truth,payload
  if (i+1)%8==0 or i==0:update(status,status='labeling',layer=layer,completed=i+1,total=len(m['rows']))
 del builder,head;gc.collect();torch.cuda.empty_cache()

@torch.no_grad()
def evaluate(head,variant,val):
 rows=[]
 for sample in val:
  d=sample['inputs'];mask=apply(d,exact(head,d).argmax(-1));y=attend(variant,sample['q'],sample['raw'],mask);truth=sample['truth'];sse=(y.float()-truth.float()).square().sum();energy=truth.float().square().sum();cost=(mask*d['costs'][None,None]).sum()
  rows.append(dict(sample_id=sample['sample_id'],nmse=float(sse/energy),sse=float(sse),energy=float(energy),keep=float(cost/(mask.shape[0]*mask.shape[1]*(d['history_tokens']+4680)))))
 return dict(nmse=sum(r['nmse']for r in rows)/len(rows),energy_weighted_nmse=sum(r['sse']for r in rows)/sum(r['energy']for r in rows),keep=sum(r['keep']for r in rows)/len(rows),worst_event_nmse=max(r['nmse']for r in rows),rows=rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--layers',nargs='*',type=int);a=p.parse_args();root,m=config(a.directory);m=dict(m,learning_rate=1e-3);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));status=root/f'strong_training/rank{rank}.json';torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch._dynamo.config.cache_size_limit=128;torch._dynamo.config.accumulated_cache_size_limit=1024
 assert json.loads((root/'preflight.json').read_text())['passed'];layers=a.layers or m['layers'][rank::world]
 for layer in layers:
  layerdir=root/f'strong_training/layer{layer:02d}';layerdir.mkdir(parents=True,exist_ok=True)
  if (layerdir/'completed.json').exists():continue
  paired_seed(layer);update(status,status='preparing',layer=layer);make_features(root,m,layer,status)
  train=[];val=[]
  for row in m['rows']:
   d=torch.load(feature_path(root,row,layer),map_location='cpu',weights_only=True);assert d['sample_id']==row['sample_id'];d['inputs']=device(d['inputs'])
   if row['split']=='train':d['labels']=device(d['labels']);d['energy']=d['energy'].cuda();train.append(d)
   else:d['q']=d['q'].cuda();d['truth']=d['truth'].cuda();d['raw']=load_raw(m['parent_experiment'],row,layer);val.append(d)
  assert len(train)==192 and len(val)==96
  initial=initial_head(m['parent_experiment'],layer);models={};opts={};curves={};best={};start=0;rng=random.Random(3000+layer)
  for variant in VARIANTS:
   model=AlignedHead(initial.state_dict()).cuda();model.load_state_dict(torch.load(root/f'training/layer{layer:02d}/{variant}/best.pt',map_location='cpu',weights_only=True)['head']);models[variant]=model;opts[variant]=torch.optim.AdamW(model.parameters(),lr=m['learning_rate'],weight_decay=.01)
  recovery=layerdir/'recovery.pt'
  if recovery.exists():
   saved=torch.load(recovery,map_location='cpu',weights_only=True);start=saved['updates'];curves=saved['curves'];best=saved['best'];rng.setstate(saved['rng'])
   for v in VARIANTS:models[v].load_state_dict(saved['heads'][v]);opts[v].load_state_dict(saved['optimizers'][v])
  else:
   for v in VARIANTS:
    result=evaluate(models[v],v,val);best[v]=result['nmse'];curves[v]=[dict(updates=0,**{k:x for k,x in result.items()if k!='rows'})];dest=layerdir/v
    state=dict(head=models[v].state_dict(),updates=0,variant=v,layer=layer,best_metric='actual_deployment_attention_nmse',best_value=best[v]);save(dest/'initial.pt',state);save(dest/'best.pt',state);atomic_json(dest/'best_validation.json',dict(updates=0,**result));atomic_json(dest/'initial_validation.json',result)
  for step in range(start+1,m['max_updates']+1):
   if (root/'STOP').exists():raise RuntimeError('STOP requested')
   chosen=rng.sample(train,m['batch_events']);losses={};lr=m['learning_rate']*min(1.,step/10)*(.1+.9*.5*(1+math.cos(math.pi*step/m['max_updates'])))
   for v in VARIANTS:
    optimizer=opts[v];optimizer.zero_grad(set_to_none=True);values=[]
    for example in chosen:
     pred=models[v](example['inputs']);temperature=.5-.25*step/m['max_updates'];loss=(pred.div(temperature).softmax(-1)*example['labels'][v]).sum()/example['energy'];values.append(loss)
    loss=torch.stack(values).mean();assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(models[v].parameters(),1.,error_if_nonfinite=True)
    for group in optimizer.param_groups:group['lr']=lr
    optimizer.step();losses[v]=dict(expected_actual_nmse=float(loss),gradient_norm=float(grad))
   if step%10==0 or step==start+1:update(status,status='training',layer=layer,updates=step,total=m['max_updates'],losses=losses,best=best,learning_rate=lr)
   if step%m['validate_every']==0:
    update(status,status='validating',layer=layer,updates=step,total=m['max_updates'])
    for v in VARIANTS:
     result=evaluate(models[v],v,val);curves[v].append(dict(updates=step,**{k:x for k,x in result.items()if k!='rows'}));dest=layerdir/v;atomic_json(dest/'curve.json',curves[v])
     if result['nmse']<best[v]:
      best[v]=result['nmse'];save(dest/'best.pt',dict(head=models[v].state_dict(),updates=step,variant=v,layer=layer,best_metric='actual_deployment_attention_nmse',best_value=best[v]));atomic_json(dest/'best_validation.json',dict(updates=step,**result))
   if step%25==0:
    save(recovery,dict(updates=step,heads={v:models[v].state_dict()for v in VARIANTS},optimizers={v:opts[v].state_dict()for v in VARIANTS},curves=curves,best=best,rng=rng.getstate()))
  for v in VARIANTS:save(layerdir/v/'last.pt',dict(head=models[v].state_dict(),updates=m['max_updates'],variant=v,layer=layer));atomic_json(layerdir/v/'curve.json',curves[v])
  atomic_json(layerdir/'completed.json',dict(layer=layer,updates=m['max_updates'],best=best,backbone_and_prototypes_frozen=True,normalization_frozen=True,initial_checkpoint_retained_as_best_if_no_improvement=True));print('completed',layer,best,flush=True)
  del train,val,models,opts,initial;gc.collect();torch.cuda.empty_cache()
 update(status,status='completed',layers=layers)
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();raise
