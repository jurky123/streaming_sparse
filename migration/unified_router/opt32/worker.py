import argparse,json,time,math,random,gc
from pathlib import Path
import torch
from .setup import ROOT,PARENT,ANCHOR,VARIANTS
from . import actions
from ..opt30.model import QueryRouter
from ..opt30.pipeline import Pipeline
from ..opt30.data import Events
from ..opt16.pipeline import GroupedRouter
from ..learning import ActionHead
from ..common import atomic_json,save,sha,paired_seed
from ..opt18.train import append

class Policy(ActionHead):
 def __init__(self,normalization):
  super().__init__(normalization)
  with torch.no_grad():self.w2.zero_();self.b2.zero_()
  self.register_buffer('penalty',torch.tensor(0.))
 def scores(self,x,gain):return gain+super().forward(x)

def load(layer):
 m=json.loads((PARENT/'experiment.json').read_text());model=QueryRouter(m,layer,ANCHOR).cuda().eval().requires_grad_(False)
 path=PARENT/f'training/layer{layer:02d}/{ANCHOR}/best.pt';manifest=json.loads((ROOT/'foundation.json').read_text());assert sha(path)==manifest['checkpoints'][str(layer)]['sha256']
 model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['model']);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False)
 return model,Events(m,layer)

@torch.no_grad()
def batch(base,events,item,kind,basis):
 q,raw,truth,targets=events.gpu(item);pipe=Pipeline(base,raw);d=actions.context(pipe,base,q,raw);a=actions.candidates(d);c=actions.content(d,base,kind,basis);x,gain=actions.features(d,a,c)
 return q,raw,truth,targets,pipe,d,a,x,gain

@torch.no_grad()
def prepare(layer):
 folder=ROOT/f'calibration/layer{layer:02d}';folder.mkdir(parents=True,exist_ok=True);base,events=load(layer);items=events.calibration_items(32000+layer,12);cov=torch.zeros(12,128,128,device='cuda');ids=[]
 for item in items[:6]:
  assert item['row']['split']=='train';q,raw,truth,targets=events.gpu(item);pipe=Pipeline(base,raw);d=actions.context(pipe,base,q,raw);a=actions.candidates(d);lab=actions.labels(q,targets,d,a);delta=lab['delta'];valid=a['valid'][lab['head'],lab['group']];weights=torch.zeros_like(valid,dtype=torch.float32)
  for typ in [1,2,3]:
   use=valid&(a['types']==typ);weights+=use/use.sum(-1,keepdim=True).clamp_min(1)/3
  cov+=torch.einsum('hna,hnad,hnae->hde',weights,delta,delta)/delta.shape[1]+torch.einsum('hnd,hne->hde',lab['base_error_vector'],lab['base_error_vector'])/delta.shape[1];ids.append(item['row']['sample_id'])
  # The generalized native metadata must reproduce the original fixed-count kernel.
  out=actions.attend(pipe,d,d['mask']);ref=pipe.attention(d['qi'],d['qs'],d['ids'],d['ix'],q.dtype);assert torch.equal(out,ref),'Base native parity'
  for index in [1,9,17]:
   if bool(a['valid'][...,index].all()):
    mask=actions.apply(d,a,torch.full(d['mask'].shape[:2],index,device='cuda',dtype=torch.long));count=int(mask[0,0].sum());assert bool((mask.sum(-1)==count).all())
    ids2=torch.where(mask,torch.arange(mask.shape[-1],device='cuda'),mask.shape[-1]).sort(-1).values[...,:count].int();candidate=actions.attend(pipe,d,mask);direct=pipe.attention(d['qi'],d['qs'],ids2,d['ix'],q.dtype);assert torch.equal(candidate,direct),'Variable-count native parity'
  atomic_json(folder/'status.json',dict(status='value_basis',events=len(ids),total=6))
 _,vectors=torch.linalg.eigh((cov+cov.transpose(1,2))*.5);basis=vectors.flip(-1)[...,:32].contiguous();normalizations={}
 for kind in ['K32','V32','V8']:
  sums=torch.zeros(12,24,device='cuda');squares=torch.zeros_like(sums);counts=torch.zeros(12,1,device='cuda')
  for item in items:
   *_,d,a,x,gain=batch(base,events,item,kind,basis);v=a['valid'][...,None].float();sums+=(x*v).sum((1,2));squares+=(x.square()*v).sum((1,2));counts+=v.sum((1,2))
  mean=sums/counts;std=(squares/counts-mean.square()).clamp_min(1e-6).sqrt();normalizations[kind]=dict(mean=mean.cpu(),std=std.cpu())
 save(folder/'parameters.pt',dict(value_basis=basis.cpu(),normalizations=normalizations,calibration_ids=[x['row']['sample_id']for x in items],value_basis_ids=ids,split='train'))
 atomic_json(folder/'completed.json',dict(completed=True,layer=layer,train_only=True,native_base_swap_add_drop_exact=True,value_basis_events=6,normalization_events=12,representation='Projected physical-page means; value basis fitted to current hard-action deltas and output errors, not V variance'))

def decide(policy,x,gain,a,options):
 logits=policy.scores(x,gain)if options['learned']else gain
 logits=logits-policy.penalty*a['cost'];valid=actions.valid_actions(a,options['mode']);logits=logits.masked_fill(~valid,-torch.inf)
 return logits,logits.argmax(-1)

@torch.no_grad()
def calibrate_penalty(base,events,policy,options,basis,items):
 if options['mode']!='dynamic':return
 grid=[0.,.0001,.0003,.001,.003,.01,.03,.1,.3,1.,3.,10.,100.];costs=torch.zeros(len(grid),device='cuda')
 for item in items:
  *_,d,a,x,gain=batch(base,events,item,options['content'],basis);logits=policy.scores(x,gain).masked_fill(~a['valid'],-torch.inf)
  for i,penalty in enumerate(grid):costs[i]+=a['cost'].gather(-1,(logits-penalty*a['cost']).argmax(-1)[...,None]).float().mean()
 chosen=next((v for v,c in zip(grid,costs.tolist())if c<=0),grid[-1]);policy.penalty.fill_(chosen)

@torch.no_grad()
def evaluate(base,events,policy,options,basis):
 rows=[]
 for item in events.val:
  q,raw,truth,targets,pipe,d,a,x,gain=batch(base,events,item,options['content'],basis);_,chosen=decide(policy,x,gain,a,options);mask=actions.apply(d,a,chosen);out=actions.attend(pipe,d,mask)[:,targets].float();h,n,_=q.shape
  execution=float(mask.sum())*d['g']*d['k']/(h*n*d['kv'].k.shape[1]);baseline=float(d['mask'].sum())*d['g']*d['k']/(h*n*d['kv'].k.shape[1])
  if options['mode']in ['swap','none']:assert torch.equal(mask.sum(-1),d['mask'].sum(-1))
  rows.append(dict(sample_id=item['row']['sample_id'],nmse=float((out-truth).square().sum()/truth.square().sum()),execution=execution,baseline_execution=baseline,acceptance=float((chosen!=0).float().mean()),add=float(((chosen>=1)&(chosen<=8)).float().mean()),drop=float(((chosen>=9)&(chosen<=16)).float().mean()),swap=float((chosen>=17).float().mean())))
 return dict(nmse=sum(x['nmse']for x in rows)/len(rows),execution=sum(x['execution']for x in rows)/len(rows),baseline_execution=sum(x['baseline_execution']for x in rows)/len(rows),acceptance=sum(x['acceptance']for x in rows)/len(rows),rows=rows)

def run(layer,name,steps):
 options=VARIANTS[name];folder=ROOT/f'training/layer{layer:02d}/{name}';folder.mkdir(parents=True,exist_ok=True);paired_seed(32000+layer);rng=random.Random(32000+layer);base,events=load(layer)
 cp=torch.load(ROOT/f'calibration/layer{layer:02d}/parameters.pt',map_location='cuda',weights_only=True);basis=cp['value_basis'];policy=Policy(cp['normalizations'][options['content']]).cuda();probes=events.calibration_items(32100+layer,6)
 calibrate_penalty(base,events,policy,options,basis,probes);initial=evaluate(base,events,policy,options,basis);best=initial['nmse']if initial['execution']<=initial['baseline_execution']+1e-6 else float('inf');curves=[dict(updates=0,**{k:v for k,v in initial.items()if k!='rows'})]
 def checkpoint(step):return dict(policy=policy.state_dict(),updates=step,options=options,anchor=ANCHOR,layer=layer,best=best,base_frozen=True,metric='actual_native_attention_nmse',backward='Expected exact original128d hard-action AV error; native selected-action NMSE in forward; not an exact native-kernel gradient')
 save(folder/'initial.pt',checkpoint(0));atomic_json(folder/'initial_validation.json',initial)
 if math.isfinite(best):save(folder/'best.pt',checkpoint(0));atomic_json(folder/'best_validation.json',dict(updates=0,**initial))
 opt=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=0.)
 for step in range(1,steps+1 if options['learned']else 1):
  opt.zero_grad(set_to_none=True);actual=expected=0.
  for item in rng.sample(events.train,2):
   with torch.no_grad():q,raw,truth,targets,pipe,d,a,x,gain=batch(base,events,item,options['content'],basis);lab=actions.labels(q,targets,d,a)
   logits,chosen=decide(policy,x,gain,a,options)
   with torch.no_grad():
    mask=actions.apply(d,a,chosen);out=actions.attend(pipe,d,mask)[:,targets].float();hard=(out-truth).square().sum()/truth.square().sum()
   sub=logits[lab['head'],lab['group']];temp=.1*(1-step/max(steps,1))+.025;prob=(sub/temp).softmax(-1);cost=a['cost'][lab['head'],lab['group']]
   surrogate=(prob*lab['errors'].detach()).sum()/truth.square().sum()+policy.penalty*(prob*cost).sum(-1).mean()
   # Report exactly the selected native output loss; use hard-action expectation for gradients.
   loss=hard+(surrogate-surrogate.detach());(loss/2).backward();actual+=float(hard)/2;expected+=float(surrogate.detach())/2
  norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);assert torch.isfinite(norm);opt.step();append(folder/'train.jsonl',dict(updates=step,actual_native_nmse=actual,backward_expected_objective=expected,gradient_norm=float(norm),penalty=float(policy.penalty)))
  atomic_json(folder/'status.json',dict(status='training',updates=step,total=steps,best=best if math.isfinite(best)else None,actual_nmse=actual))
  if step%100==0 or step==steps:
   calibrate_penalty(base,events,policy,options,basis,probes);val=evaluate(base,events,policy,options,basis);curves.append(dict(updates=step,**{k:v for k,v in val.items()if k!='rows'}));atomic_json(folder/'curve.json',curves)
   if val['execution']<=val['baseline_execution']+1e-6 and val['nmse']<best:best=val['nmse'];save(folder/'best.pt',checkpoint(step));atomic_json(folder/'best_validation.json',dict(updates=step,**val))
   save(folder/'last.pt',checkpoint(step));print(name,layer,step,val['nmse'],val['execution'],flush=True)
 # Benchmark complete current-KV route+features+decision+native, cached history; no duplicate base attention.
 if (folder/'best.pt').exists():policy.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['policy'])
 from ..opt5.probe import capture,timing
 with torch.no_grad():
  item=next(x for x in events.val if x['row']['history_tokens']==14040 and x['row']['step']==3);q,raw,truth,targets=events.gpu(item);pipe=Pipeline(base,raw)
  def forward():
   d=actions.context(pipe,base,q,raw)
   if options['mode']=='none':return pipe.attention(d['qi'],d['qs'],d['ids'],d['ix'],q.dtype)
   a=actions.candidates(d);x,gain=actions.features(d,a,actions.content(d,base,options['content'],basis));_,chosen=decide(policy,x,gain,a,options);return actions.attend(pipe,d,actions.apply(d,a,chosen))
  reference=forward();graph,out=capture(forward);graph.replay();torch.cuda.synchronize();assert torch.equal(reference,out);ms=timing(graph,100)
 atomic_json(folder/'completed.json',dict(completed=True,layer=layer,variant=name,updates=steps if options['learned']else 0,best_native_nmse=best if math.isfinite(best)else None,budget_feasible=math.isfinite(best),module_ms=ms,scope='480p H14040 step3 warm complete module including summary/MLP/actions; correctness reference implementation, no claimed kernel optimization',options=options))
 atomic_json(folder/'status.json',dict(status='completed',updates=steps if options['learned']else 0,total=steps,best=best if math.isfinite(best)else None))

def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--variant');p.add_argument('--prepare',action='store_true');p.add_argument('--steps',type=int,default=1200);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 if a.prepare:
  with torch.no_grad():prepare(a.layer)
 else:run(a.layer,a.variant,a.steps)
if __name__=='__main__':main()
