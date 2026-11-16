"""Fixed E43 completion: active-group Adam, original loss ablations, beta contrast."""
from ..e42_local_coverage.core import *
from ..e29_local_reader.core import page_features
from ..e25_decision_diagnostics.run import conditional_metrics
from ..e28_local_response.run import digest
E43=ROOT/'artifacts/e43_fit_v1';D=ROOT/'artifacts/e44_fit_diagnostics_v1'
ARMS=['L_joint_sparse','L_physical','L_set','P_physical','P_set','P_contrast']
def protocol(seed):
 dest=D/f's{seed}';dest.mkdir(parents=True,exist_ok=True)
 spec=dict(seed=seed,events=json.loads((E43/f's{seed}/protocol.json').read_text())['events'],updates=2304,states=96,arms=ARMS,lr=1e-4,Adam='reset, betas .9/.999 eps1e-8; no weight decay',L_update='one parameter and Adam optimizer per trajectory/layer/head; only active optimizer is stepped, 96 visits each group',fairness_boundary='Sparse clocks correct inactive Adam aging; they do not equate P/L parameterization or update amplitude. No lr multiplier.',losses='joint original; physical=local+quality; set=setloss; contrast=local+group-quality+within-R/A-quality+original set',contrast='uniform detached page weights; four role means, within R/A residuals; lambda_group=lambda_contrast=1 fixed; normalization original per-page bs then group mean bs',checkpoints=[0,384,768,1152,1536,1920,2304],final=2304,layer29_teacher=['M16B16','M16B128','Q64B16','Q64B128'],Test=False,raw_KV='offline identity verification only; no raw Attention',saturation='descriptive frozen phi; no feature change')
 dump(dest/'protocol.json',spec);return spec

def build(seed,verify=True):
 data=Data(seed);spec=protocol(seed);model=model_from(E29/f's{seed}/selected.pt');states=[];groups={};identity={};initial=[];checks=[];idrecords=[]
 for event in spec['events']:
  d=data.load(event);m=d['meta'];key=(m['trajectory'],m['layer'],m['head']);gid=str(key);sx=torch.load(CACHE/f'train_s{seed}'/m['summary_file'],map_location='cpu',weights_only=True);co=model.page(page_features(d['s'])).detach();ids=[]
  raw=None
  if verify:
   raw=torch.load(OUT/sx['kv_file'],map_location='cpu',weights_only=True)
  for p,h in enumerate(sx['summary_hashes']):
   actual=digest(*(getattr(d['s'],n)[0,p] for n in ['keys','mean','detail','stats']));assert actual==h
   rh=digest(raw['ids'][p],raw['k'][p],raw['v'][p]) if raw is not None else None
   if p<sx['history_pages'] and verify:assert rh==sx['raw_hashes'][p]
   ik=(key,h)
   if ik not in identity:identity[ik]=len(initial);initial.append(co[p])
   ids.append(identity[ik]);idrecords.append(dict(event=event,page=p,group=gid,coefficient_id=identity[ik],summary_sha=h,raw_sha=rh,history=p<sx['history_pages']))
  d['ids']=torch.tensor(ids,device='cuda');d['group']=gid;states.append(d)
  C=torch.zeros_like(d['mask']);C[d['r']]=True;C[d['a']]=True;rr=torch.where(d['mask']&~C)[0];rr=rr[rr!=d['own']];aa=torch.where(~d['mask']&~C)[0];mask=d['mask'].clone();mask[rr[0]]=False;mask[aa[0]]=True;st=dict(mask=mask.cpu().tolist(),remove=d['r'].cpu().tolist(),add=d['a'].cpu().tolist());bg=data.load(event,st);bg.update(ids=d['ids'],group=gid);states.append(bg)
  if gid not in groups:groups[gid]=d['ids']
  else:assert torch.equal(groups[gid],d['ids'])
 # No summary-equivalent/raw-distinct collisions are silently accepted.
 coll=collections.defaultdict(set)
 for r in idrecords:
  if r['raw_sha'] is not None:coll[r['coefficient_id']].add(r['raw_sha'])
 collisions=sum(len(v)>1 for v in coll.values());assert collisions==0,collisions
 table(D/f's{seed}/identities.csv',idrecords);dump(D/f's{seed}/identity_audit.json',dict(passed=True,records=len(idrecords),physical_pages=len(initial),summary_raw_collisions=collisions,groups=len(groups),raw_attention=False))
 return data,states,groups,torch.stack(initial)

class Lookup(torch.nn.Module):
 def __init__(self,groups,initial):
  super().__init__();self.group_names=list(groups);self.group_index={g:i for i,g in enumerate(groups)};self.co=torch.nn.ParameterList([torch.nn.Parameter(initial[groups[g]].clone()) for g in groups])
 def forward(self,d):
  co=self.co[self.group_index[d['group']]].reshape(-1,5,17).bfloat16().float();v=co[None,:,0]+torch.einsum('qpj,pjk->qpk',d['F'][...,1:].float(),co[:,1:]);return v[...,:16],v[...,16]
 def active(self,d):return [self.co[self.group_index[d['group']]]]

def contrast_quality(bp,bt,bs,d):
 R=torch.unique(d['r']);A=torch.unique(d['a']);cand=torch.zeros_like(d['mask']);cand[R]=True;cand[A]=True;roles=[R,A,torch.where(d['mask']&~cand)[0],torch.where(~d['mask']&~cand)[0]];means=[];within=[]
 for j,ix in enumerate(roles):
  if not len(ix):continue
  pred=bp[:,ix];true=bt[:,ix];pm=pred.mean(-1,keepdim=True);tm=true.mean(-1,keepdim=True);scale=bs[:,ix].mean(-1,keepdim=True);means.append(torch.nn.functional.smooth_l1_loss(pm/scale,tm/scale))
  if j<2:within.append(torch.nn.functional.smooth_l1_loss((pred-pm)/bs[:,ix],(true-tm)/bs[:,ix]))
 return torch.stack(means).mean()+torch.stack(within).mean()

def outputs(model,d):return model(d) if isinstance(model,Lookup) else model(d['s'],d['F'])
def terms(model,d,data,arm):
 rp,bp=outputs(model,d);ls,bs,gs,_,_=losses(rp,bp,d['rtrue'],d['btrue'],d['mass'],d['lm'],d['center'],d['mask'],d['r'],d['a'],d['g'],d['norm'],data.scale,data.fit)
 if arm.endswith('physical'):loss=ls[0]+ls[1]
 elif arm.endswith('_set'):loss=ls[2]
 elif arm=='P_contrast':loss=ls[0]+contrast_quality(bp,d['btrue'],bs,d)+ls[2]
 else:loss=sum(ls)
 return loss,ls,rp,bp

@torch.no_grad()
def evaluate(models,states,seed,step):
 rows=[];arrays={};conditions=[]
 for name,model in models.items():
  pp=[];tt=[]
  for i,d in enumerate(states):
   rp,bp=outputs(model,d);sc,ind=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);pp.append(sc.cpu().numpy());tt.append(d['g'].cpu().numpy());R=torch.unique(d['r']);A=torch.unique(d['a']);ce=[]
   for ix in [R,A]:
    err=bp[:,ix]-d['btrue'][:,ix];ce.append(float((err-err.mean(-1,keepdim=True)).abs().mean()))
   for policy,scores in [('Set',sc),('Independent',ind)]:
    j=int(scores.argmax());ex=bool(scores[j]>0);rows.append(dict(seed=seed,step=step,arm=name,state=i,event=d['meta']['event'],layer=d['meta']['layer'],trajectory=d['meta']['trajectory'],policy=policy,initial=d['initial'],oracle_gain=max(0.,float(d['g'].max())),executed_gain=float(d['g'][j]) if ex else 0.,execute=ex,proposal_gain=float(d['g'][j]),hindsight_gain=max(0.,float(d['g'][j])),action=j if ex else -1,local_mse=float((rp-d['rtrue']).square().mean()),local_mass_mse=float((d['mass'][...,None]*(rp-d['rtrue'])).square().mean()),beta_mae=float((bp-d['btrue']).abs().mean()),beta_contrast_mae=float(np.mean(ce))))
  p=np.stack(pp);t=np.stack(tt);arrays[name]=p
  for i in range(48):conditions.append(dict(seed=seed,step=step,arm=name,event=states[2*i]['meta']['event'],layer=states[2*i]['meta']['layer'],**conditional_metrics(t[2*i:2*i+2],p[2*i:2*i+2])))
 arrays['true_gain']=np.stack(tt);np.savez_compressed(D/f's{seed}/scores_{step}.npz',**arrays);table(D/f's{seed}/evaluation_{step}.csv',rows);table(D/f's{seed}/conditional_{step}.csv',conditions)
 return rows

def run(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);data,states,groups,initial=build(seed);models={name:Lookup(groups,initial).cuda() if name.startswith('L') else model_from(E29/f's{seed}/selected.pt') for name in ARMS};opts={}
 for name,m in models.items():opts[name]=[torch.optim.Adam([p],lr=1e-4,betas=(.9,.999),eps=1e-8) for p in m.co] if isinstance(m,Lookup) else torch.optim.Adam(m.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8)
 # Original frozen E43 models for complete matrix audit, initialized lookup matches original global ordering.
 baseP=model_from(E43/f's{seed}/P_final.pt');baseL=Lookup(groups,initial).cuda();old=torch.load(E43/f's{seed}/L_final.pt',map_location='cuda',weights_only=True)['co']
 for gid,ids in groups.items():baseL.co[baseL.group_index[gid]].data.copy_(old[ids])
 evaluate(dict(E43_P=baseP,E43_L=baseL),states,seed,-1);initrows=evaluate(models,states,seed,0)
 for i in range(96):
  vals=[r['executed_gain'] for r in initrows if r['state']==i and r['policy']=='Set'];assert max(vals)-min(vals)<1e-8
 curves=[];start=time.time()
 for u in range(2304):
  d=states[u%96]
  for name,m in models.items():
   loss,ls,rp,bp=terms(m,d,data,name);opt=opts[name][m.group_index[d['group']]] if isinstance(m,Lookup) else opts[name];pars=m.active(d) if isinstance(m,Lookup) else list(m.parameters());opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(pars,1,error_if_nonfinite=True);opt.step();curves.append(dict(update=u,arm=name,local=float(ls[0]),quality=float(ls[1]),setloss=float(ls[2]),total=float(loss),gradnorm=float(gn)))
  if (u+1)%384==0:evaluate(models,states,seed,u+1);table(D/f's{seed}/curves.csv',curves);print('STEP',seed,u+1,round(time.time()-start),flush=True)
 for name,m in models.items():torch.save(m.state_dict(),D/f's{seed}/{name}_final.pt')
 steps={}
 for name,m in models.items():
  if isinstance(m,Lookup):
   v=[float(next(iter(o.state.values()))['step']) for o in opts[name]];assert set(v)=={96.};steps[name]=v
 data.check_frozen();dump(D/f's{seed}/complete.json',dict(seed=seed,updates=2304,sparse_group_steps=steps,groups=len(groups),states=96,test=False,original_states_reused=True,parameters={n:sum(p.numel() for p in m.parameters()) for n,m in models.items()}));print('DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
