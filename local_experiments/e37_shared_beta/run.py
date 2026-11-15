from ..e30_beta_diagnostics.run import *
from ..e25_decision_diagnostics.report import aggregate
import csv
E30=DEST
DEST=ROOT/'artifacts/e37_shared_beta_v1'
METHODS=['E29','Shared-Q','CandidateBackground-Q','TrueBeta']
def fit(lm,d,idx):
 w=(2*lm[:,idx]).softmax(1);return (w*d[:,idx]).sum(1)
@torch.no_grad()
def run(seed):
 torch.set_num_threads(2);dest=DEST/f's{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'complete.json').exists();qs=torch.linspace(0,63,16,device='cuda').round().long();rows=[];quality=[];conditions=[];decomp=[];offsets=[];start=time.time();source=[]
 base={r['event']:r for r in csv.DictReader((E30/'decision_per_event.csv').open()) if int(r['seed'])==seed and r['method']=='Predicted' and r['policy']=='Set'}
 for idx,path in enumerate(files('development')):
  xp=E30/f's{seed}/beta_{idx:04d}.pt';pp=E30/f's{seed}/panel_{idx:04d}.npz';x=torch.load(xp,map_location='cuda',weights_only=True);p=np.load(pp);event=str(path.relative_to(OUT));assert event==x['event']==str(p['event']);ev=torch.load(path,weights_only=True);norm=(ev['dense'][qs.cpu()].square().sum()/16+1e-12).cuda();meta=dict(seed=seed,event=event,**{k:ev[k] for k in ['trajectory','layer','chunk','step','head','tile']});J=float(base[event]['initial']);mask=torch.as_tensor(p['states'][0],device='cuda');r=torch.as_tensor(p['initial_remove'],device='cuda');a=torch.as_tensor(p['initial_add'],device='cuda');g=p['true_initial_gain'];oracle=max(0.,float(g.max()));C=torch.zeros(len(mask),device='cuda',dtype=torch.bool);C[r]=True;C[a]=True;assert int(C.sum())==48
  lm=x['logmass'][qs].double();pred=x['betas']['Predicted'][qs].double();true=x['beta_true'][qs].double();d=true-pred;allidx=torch.ones_like(C);b=fit(lm,d,allidx);bc=fit(lm,d,C);bb=fit(lm,d,~C);correction=torch.where(C[None],bc[:,None],bb[:,None]);betas={'E29':pred,'Shared-Q':pred+b[:,None],'CandidateBackground-Q':pred+correction,'TrueBeta':true}
  w=(2*lm).exp();s0=(w*d.square()).sum(1);s1=(w*(d-b[:,None]).square()).sum(1);s2=(w*(d-correction).square()).sum(1);assert torch.all(s2<=s1+1e-9) and torch.all(s1<=s0+1e-9)
  for j,q in enumerate(qs.tolist()):
   decomp.append(dict(**meta,query=q,weight_sum=float(w[j].sum()),total_sse=float(s0[j]),shared_explained=float(s0[j]-s1[j]),candidate_background_extra=float(s1[j]-s2[j]),within_group_sse=float(s2[j]),shared_fraction=float((s0[j]-s1[j])/s0[j]) if s0[j]>0 else 0.,two_fraction=float((s0[j]-s2[j])/s0[j]) if s0[j]>0 else 0.));offsets.append(dict(**meta,query=q,shared=float(b[j]),candidate=float(bc[j]),background=float(bb[j])))
  initial={};common={};inds={}
  for name,beta in betas.items():
   def score(mm,rr,aa):return local_scores(x['predicted_response'][qs],beta,lm,x['center'][qs],mm,rr,aa,norm)
   sc,ind=score(mask,r,a);initial[name]=sc.cpu().numpy();inds[name]=ind.cpu().numpy()
   for policy,scores in [('Set',sc),('Independent',ind)]:
    k=int(scores.argmax());execute=bool(scores[k]>0);gain=float(g[k]) if execute else 0.;new=mask.clone()
    if execute:new[r[k]]=False;new[a[k]]=True
    assert new.sum()==mask.sum() and new[ev['own']]
    rows.append(dict(**meta,method=name,policy=policy,initial=J,oracle_gain=oracle,proposal_gain=float(g[k]),executed_gain=gain,hindsight_gain=max(0.,float(g[k])),execute=execute,remove=int(r[k]),add=int(a[k]),predicted_gain=float(scores[k]),final_nmse=J-gain))
   for group,ix in [('all',allidx),('candidate',C),('background',~C)]:
    er=beta[:,ix]-true[:,ix];ww=w[:,ix];weighted_bias=(ww*er).sum()/ww.sum();mse=(ww*er.square()).sum()/ww.sum();var=(ww*(er-weighted_bias).square()).sum()/ww.sum();quality.append(dict(**meta,method=name,group=group,beta_mae=float(er.abs().mean()),beta_rmse=float(er.square().mean().sqrt()),weighted_bias=float(weighted_bias),weighted_mse=float(mse),centered_variance=float(var),weight_sum=float(ww.sum()),weighted_sse=float((ww*er.square()).sum()),gain_spearman=float(spearmanr(initial[name],g).statistic),gain_sign_accuracy=float(((initial[name]>0)==(g>1e-12)).mean())))
   rr=torch.as_tensor(p['remove'],device='cuda');aa=torch.as_tensor(p['add'],device='cuda');ss=np.stack([score(torch.as_tensor(m,device='cuda'),rr,aa)[0].cpu().numpy() for m in p['states']]);common[name]=ss;conditions.append(dict(**meta,method=name,**conditional_metrics(p['true_gain'],ss)))
  assert np.allclose(initial['E29'],p['initial_Predicted'],rtol=1e-9,atol=1e-10) and np.allclose(initial['TrueBeta'],p['initial_True'],rtol=1e-9,atol=1e-10)
  np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=event,states=p['states'],remove=p['remove'],add=p['add'],initial_remove=p['initial_remove'],initial_add=p['initial_add'],true_gain=p['true_gain'],true_initial_gain=g,**common,**{'initial_'+k:v for k,v in initial.items()},**{'independent_'+k:v for k,v in inds.items()},shared=b.cpu().numpy(),candidate=bc.cpu().numpy(),background=bb.cpu().numpy());source.append(dict(event=event,beta_sha=sha(xp),panel_sha=sha(pp),event_sha=sha(path)))
  if (idx+1)%48==0:print(seed,idx+1,round(time.time()-start),flush=True)
 for name,data in [('decisions',rows),('quality',quality),('conditional',conditions),('decomposition',decomp),('offsets',offsets),('sources',source)]:
  for row in data:
   for k,v in row.items():
    if isinstance(v,float) and not math.isfinite(v):row[k]=None
  dump(dest/(name+'.json'),data)
 dump(dest/'complete.json',dict(seed=seed,events=288,seconds=time.time()-start,protocol_sha=sha(DEST/'protocol.json'),training=False,test=False,raw_kv_loaded=False));print('DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);run(p.parse_args().seed)
