from ..e30_beta_diagnostics.run import *
from ..e25_decision_diagnostics.report import aggregate
from ..e25_decision_diagnostics.run import project_gains
E30=DEST
DEST=ROOT/'artifacts/e36_query_coverage_v1'
A_METHODS=['E29-M16','E29-Q64','True-M16','True-Q64','TrueBeta-Q64']
B_METHODS=['E29-Coreset16','True-Coreset16']

def coreset(q):
 q=q.double();first=int((q-q.mean(0)).square().sum(-1).argmin());chosen=[];dist=[];nearest=torch.full((len(q),),torch.inf,device=q.device,dtype=q.dtype)
 for k in range(16):
  j=first if k==0 else int(nearest.argmax());chosen.append(j);dd=(q-q[j]).square().sum(-1);dist.append(dd);nearest=torch.minimum(nearest,dd);nearest[chosen]=-torch.inf
 ids=torch.tensor(chosen,device=q.device);D=torch.stack(dist,1);assignment=D.argmin(1);counts=torch.bincount(assignment,minlength=16);return ids,counts.double()/64,assignment

def pred_score(x,qs,mask,r,a,norm,truthbeta=False,weights=None):
 rp=x['predicted_response'][qs].double();center=x['center'][qs].double();beta=(x['beta_true'] if truthbeta else x['betas']['Predicted'])[qs];lm=x['logmass'][qs]
 if weights is not None:
  keep=weights>0;rp=rp[keep];center=center[keep];beta=beta[keep];lm=lm[keep];w=weights[keep];factor=(len(w)*w).sqrt()[:,None,None];rp=rp*factor;center=center*factor
 return local_scores(rp,beta,lm,center,mask,r,a,norm)

def true_weighted(c,d,mask,r,a,qs,weights):
 N,Z,od=d['N'][qs],d['Z'][qs],d['dense'][qs];nh,zh=c['nh'][qs],c['zh'][qs];ns=torch.where(mask[None,:,None],N,nh).sum(1);zs=torch.where(mask[None],Z,zh).sum(1);E=od-ns/zs[:,None];den=(weights*od.square().sum(-1)).sum()+1e-12/64;out=[]
 for i in range(0,len(r),64):
  rr=r[i:i+64];aa=a[i:i+64];newN=ns[:,None]-(N-nh)[:,rr]+(N-nh)[:,aa];newZ=zs[:,None]-(Z-zh)[:,rr]+(Z-zh)[:,aa];delta=newN/newZ[...,None]-ns[:,None]/zs[:,None,None];out.append(((2*E[:,None]*delta-delta.square()).sum(-1)*weights[:,None]).sum(0)/den)
 return torch.cat(out)

@torch.no_grad()
def run(seed,stage='A',limit=None):
 torch.set_num_threads(2);dest=DEST/f'{stage}_s{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'complete.json').exists()
 if stage=='B':assert json.loads((DEST/'A_gate.json').read_text())['passed']
 bundle=bundle_for('C',seed);paths=files('development');uni=torch.linspace(0,63,16,device='cuda').round().long();allq=torch.arange(64,device='cuda');rows=[];tops=[];quality=[];conditional=[];audit=[];start=time.time()
 for idx,path in enumerate(paths):
  if limit is not None and idx>=limit:break
  d,s,c,_,_,_,_=prepare_local(bundle,path);x=torch.load(E30/f's{seed}/beta_{idx:04d}.pt',map_location='cuda',weights_only=True);p=np.load(E30/f's{seed}/panel_{idx:04d}.npz');event=str(path.relative_to(OUT));assert x['event']==event==str(p['event']);mask=torch.as_tensor(p['states'][0],device='cuda');r=torch.as_tensor(p['initial_remove'],device='cuda');a=torch.as_tensor(p['initial_add'],device='cuda');g=torch.as_tensor(p['true_initial_gain'],device='cuda');J=float(objective(c,d['N'],d['Z'],d['dense'],mask));meta=dict(seed=seed,event=event,**{k:d['ev'][k] for k in ['trajectory','layer','chunk','step','head','tile']});oracle=max(0.,float(g.max()));selection={}
  if stage=='A':configs=[(n,uni if n.endswith('M16') else allq,None) for n in A_METHODS]
  else:
   ids,w,assignment=coreset(d['q']);configs=[(n,ids,w) for n in B_METHODS];selection=dict(ids=ids.cpu().numpy(),weights=w.cpu().numpy(),assignment=assignment.cpu().numpy());assert len(torch.unique(ids))==16 and abs(float(w.sum())-1)<1e-12
  initial={};common={};independent={}
  for method,qs,weights in configs:
   teacher=method.startswith('True-');norm=d['dense'][qs].square().sum()/len(qs)+1e-12 if weights is None else (weights*d['dense'][qs].square().sum(-1)).sum()+1e-12
   def scores(mm,rr,aa):
    if teacher:return (project_gains(c,d,mm,rr,aa,qs) if weights is None else true_weighted(c,d,mm,rr,aa,qs,weights)),None
    return pred_score(x,qs,mm,rr,aa,norm,truthbeta=method.startswith('TrueBeta'),weights=weights)
   sc,ind=scores(mask,r,a);assert torch.isfinite(sc).all();initial[method]=sc.cpu().numpy()
   if ind is not None:independent[method]=ind.cpu().numpy()
   for policy,score in [('Set',sc)]+([('Independent',ind)] if ind is not None else []):
    j=int(score.argmax());execute=bool(score[j]>0);new=mask.clone()
    if execute:new[r[j]]=False;new[a[j]]=True
    assert int(new.sum())==int(mask.sum()) and bool(new[d['own']]);gain=float(g[j]) if execute else 0.
    rows.append(dict(**meta,method=method,policy=policy,queries=len(qs),initial=J,oracle_gain=oracle,proposal_gain=float(g[j]),hindsight_gain=max(0.,float(g[j])),executed_gain=gain,execute=execute,remove=int(r[j]),add=int(a[j]),predicted_gain=float(score[j]),absolute_nmse_increase=-gain,final_nmse=J-gain))
   oo=np.argsort(-sc.cpu().numpy(),kind='stable');truth=g.cpu().numpy()
   for L in [1,4,8]:
    values=truth[oo[:L]];best=max(0.,float(values.max()));tops.append(dict(**meta,method=method,L=L,initial=J,oracle_gain=oracle,best_gain=best,has_positive=bool(values.max()>1e-12),regret=oracle-best,best_hit=best>=oracle-1e-12))
   quality.append(dict(**meta,method=method,pearson=stablecorr(initial[method],truth),spearman=float(spearmanr(initial[method],truth).statistic),sign_accuracy=float(((sc>0)==(g>1e-12)).double().mean())))
   rr=torch.as_tensor(p['remove'],device='cuda');aa=torch.as_tensor(p['add'],device='cuda');ss=np.stack([scores(torch.as_tensor(st,device='cuda'),rr,aa)[0].cpu().numpy() for st in p['states']]);common[method]=ss;conditional.append(dict(**meta,method=method,**conditional_metrics(p['true_gain'],ss)))
  if stage=='A':
   assert np.allclose(initial['E29-M16'],p['initial_Predicted'],rtol=1e-9,atol=1e-10);assert np.allclose(initial['True-Q64'],p['true_initial_gain'],rtol=1e-6,atol=1e-9)
  np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=event,states=p['states'],remove=p['remove'],add=p['add'],initial_remove=p['initial_remove'],initial_add=p['initial_add'],true_initial_gain=truth,true_gain=p['true_gain'],**common,**{'initial_'+k:v for k,v in initial.items()},**{'independent_'+k:v for k,v in independent.items()},**selection)
  audit.append(dict(**meta,stage=stage,mask_pool_unchanged=True))
  if (idx+1)%24==0:
   for name,data in [('decisions',rows),('topL',tops),('quality',quality),('conditional',conditional),('audit',audit)]:
    for row in data:
     for k,v in row.items():
      if isinstance(v,float) and not math.isfinite(v):row[k]=None
    dump(dest/(name+'.json'),data)
   print('EVENTS',stage,seed,idx+1,round(time.time()-start),flush=True)
 for name,data in [('decisions',rows),('topL',tops),('quality',quality),('conditional',conditional),('audit',audit)]:dump(dest/(name+'.json'),data)
 if limit is None:dump(dest/'complete.json',dict(seed=seed,stage=stage,events=288,seconds=time.time()-start,C_sha=sha(checkpoint('C',seed)),reader_sha=sha(E29/f's{seed}/selected.pt'),protocol_sha=sha(DEST/'protocol.json'),training=False,test_accessed=False,new_QKV=False))
 print('DONE',stage,seed,len(audit),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',choices=['A','B'],default='A');p.add_argument('--limit',type=int);a=p.parse_args();run(a.seed,a.stage,a.limit)
