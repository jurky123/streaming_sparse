from ..e20_compressed_response.core import *
from ..e12_swap_prediction.core import pool,swap_labels,objective,split
import argparse
OUT=ROOT/'artifacts/e23_dual_representatives_v1'
def setup():
 OUT.mkdir(exist_ok=True)
 if (OUT/'protocol.json').exists():return
 dump(OUT/'protocol.json',dict(created=time.time(),scope='E23A all144 frozen events initial Risk only; Q64/Page64/R4/rank8, M16 scoring, unchanged summary/Risk/Medium one swap. No raw KV access, new training, regrouping or new permanent per-page storage.',R2='fixed adjacent representative pairs(0,1),(2,3); arithmetic mean K and decoded V per pair, token mass32. R2 logits are pairwise means of EXISTING R4 logits; not another QK scan. Not mass-averaging R4 exponentials (which would reproduce R4 Z). No pair search.',proxy='deltaN=N4-N2,deltaZ=Z4-Z2 in common all-R4-Z scale; assume Raw extrapolation=2R4-R2, factor1 frozen, no fitted sign/scale. e_on=deltaN-O4all*deltaZ. EstimatedDense=O4all+sum(e_on)/(1+sum(deltaZ)); net proxy=e_on-(estimatedDense-O4all)*deltaZ so approximate mixed objective is internally consistent. Online-centered e_on is not mathematically exact Dense-centered residual.',methods='primary dual_rank8 projected onto orthonormal current per-head basis, full128 dual_full capacity/cost diagnostic; baseline existing FullV centered contribution projected rank8, d=0 (value_rank8). Each Set and Independent same information, same Medium one-action/no-op. Independent mean squared net page residual divided by estimatedDenseZ^2, score add-remove.',splits='36city101 train descriptive;36city303 validation decision;72horse evaluation frozen, no selection. No fitting in A. Run train/validation first, lock stageB decision before horse.',gate='Primary dual_rank8 Set: validation mean true relative improvement>0 and degradation<=15%, AND greater mean improvement than same-information Independent and existing value_rank8 Set. If fails B not run. If passes B limited raw-snapshot feasibility, no claim144 can retrain summaries withoutrawKV. Full128 diagnostic cannot override gate.',metrics='Q64 real NMSE vs unchanged old Risk, signed e_on correlation/NMSE, net residual cosine, I sign/B error matched16, swap ranking/regret, cost accounting. Teachers evaluation only.',baseline_sha=sha(DATA/'manifest.json')))
def run(eid,device):
 dest=OUT/'events'/eid;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return
 f=dict(np.load(DATA/'contributions'/eid/'summary_and_route_features.npz'));raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));T=lambda x:torch.as_tensor(x,device=device,dtype=torch.float64);qs=f['probe_indices'].astype(int);q=T(f['query'][qs]);kap=T(f['summary_keys']);basis=T(f['basis']);U=torch.linalg.qr(basis).Q;haar=T([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]);nu=T(f['summary_mean'])[:,None]+torch.einsum('ag,pak,dk->pgd',haar,T(f['summary_detail']),basis);ell=torch.einsum('qd,pgd->qpg',q,kap)/math.sqrt(128);scale=(ell+math.log(16)).flatten(1).logsumexp(1);w4=16*(ell-scale[:,None,None]).exp();z4=w4.sum(-1);n4=torch.einsum('qpg,pgd->qpd',w4,nu);o4=n4.sum(1);P=kap.shape[0];ell2=ell.reshape(16,P,2,2).mean(-1);nu2=nu.reshape(P,2,2,128).mean(2);w2=32*(ell2-scale[:,None,None]).exp();z2=w2.sum(-1);n2=torch.einsum('qpg,pgd->qpd',w2,nu2);dz=z4-z2;dn=n4-n2;eo=dn-o4[:,None]*dz[...,None];od=o4+eo.sum(1)/(1+dz.sum(1))[:,None];en=eo-(od-o4)[:,None]*dz[...,None];zp=z4+dz;assert (dz>=-1e-12).all()
 # Teacher coordinate conversion exclusively for diagnostics.
 fac=1/T(raw['summary_mass'][qs]).sum(1);et=T(raw['e'][qs])*fac[:,None,None];dt=T(raw['d'][qs])*fac[:,None];teacher_on=et+(T(raw['dense_output'][qs])-o4)[:,None]*dt[...,None];zt=T(raw['exact_mass'][qs])*fac[:,None];r,a=pool({'risk':f['risk_scores']},raw['risk'],int(raw['own']));truth=swap_labels(raw,raw['risk'],r,a);J=objective(raw,raw['risk']);_,_,trueparts=gain_scores(et,zt,z4,raw['risk'],r,a,np.arange(16),True);h=n4-o4[:,None]*z4[...,None];models={'dual_rank8':(en@U,zp,eo@U,teacher_on@U,et@U),'dual_full':(en,zp,eo,teacher_on,et),'value_rank8':(h@U,z4,h@U,teacher_on@U,et@U)};rows=[];diag=[];saved={};checks=[]
 for name,(e,z,online,target,nettarget) in models.items():
  score,bad,parts=gain_scores(e,z,z4,raw['risk'],r,a,np.arange(16),True);assert score is not None and bad==0
  def nmse(x,y):return float((x-y).square().sum()/(y.square().sum()+1e-30))
  def cos(x,y):return float(((x*y).sum(-1)/(x.norm(dim=-1)*y.norm(dim=-1)+1e-30)).mean())
  valid=abs(trueparts['I'])>1e-12;diag.append(dict(event_id=eid,split=split(eid),method=name,eon_nmse=nmse(online,target),eon_cosine=cos(online,target),eon_pearson=float(np.corrcoef(array(online).flatten(),array(target).flatten())[0,1]),net_nmse=nmse(e,nettarget),net_cosine=cos(e,nettarget),I_sign_accuracy=float(np.mean((parts['I'][valid]>0)==(trueparts['I'][valid]>0))),B_nmse=float(np.sum((parts['B']-trueparts['B'])**2)/(np.sum(trueparts['B']**2)+1e-30))))
  page=(e.square().sum(-1)/z.sum(1)[:,None].square()).mean(0);ind=array(page)[a]-array(page)[r]
  for policy,s in [('set',score),('independent',ind)]:
   met=metrics(s[None],truth[None],np.array([J]));j=int(s.argmax());x=raw['risk'].copy();execute=bool(s[j]>0)
   if execute:x[r[j]]=False;x[a[j]]=True
   jnew=objective(raw,x);assert x.sum()==int(raw['K']) and x[int(raw['own'])];error=abs(J-jnew-(truth[j] if execute else 0));assert error<1e-10;checks.append(error);rows.append(dict(event_id=eid,split=split(eid),method=name,policy=policy,initial_nmse=J,final_nmse=jnew,remove=int(r[j]),add=int(a[j]),execute=execute,**met));saved[name+'_'+policy]=s
 # Full128 exchange consistency: explicit extrapolated N/Z mixed-output objective.
 nstar=n4+dn;sc=saved['dual_full_set']
 def loss(x):
  ns=torch.where(x[None,:,None],nstar,n4).sum(1);zs=torch.where(x[None],zp,z4).sum(1);return (od-ns/zs[:,None]).square().sum(-1).mean()
 initial=torch.as_tensor(raw['risk'],device=device);err=0
 for j in [0,len(r)-1,int(sc.argmax())]:
  x=initial.clone();x[r[j]]=False;x[a[j]]=True;err=max(err,abs(float(loss(initial)-loss(x))-sc[j]))
 assert err<1e-10
 np.savez_compressed(dest/'scores.npz',**saved,true_gain=truth,remove=r,add=a);dump(dest/'result.json',dict(rows=rows,diagnostics=diag,audit=dict(direct_extrapolated_objective_error=err,selected_true_gain_error=max(checks),online_sha=sha(DATA/'contributions'/eid/'summary_and_route_features.npz'),teacher_sha=sha(DATA/'contributions'/eid/'residuals_fp64.npz'))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--phase',choices=['development','evaluation']);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.setup:setup()
 else:
  ids=sorted(p.name for p in (DATA/'contributions').iterdir() if (split(p.name)=='evaluation')==(a.phase=='evaluation'))
  for eid in ids[a.worker::a.workers]:run(eid,'cuda:'+str(a.worker))
  print(a.phase,a.worker,'done',flush=True)
