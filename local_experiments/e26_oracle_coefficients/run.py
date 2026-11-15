from .core import *

@torch.no_grad()
def run(seed,limit=None):
 torch.set_num_threads(2);dest=E26/(f'pilot_s{seed}' if limit else f's{seed}');dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 bc=bundle_for('C',seed);bd=bundle_for('D',seed);fit=torch.load(OUT/'fit.pt',weights_only=True);B=fit['B'].cuda();qs=torch.linspace(0,63,16,device='cuda').round().long();paths=files('development');rows=[];conds=[];fits=[];audits=[];failures=[];start=time.time()
 indices=list(range(len(paths))) if limit is None else [0,47,95,143,191,239,287][:limit]
 for idx in indices:
  path=paths[idx];d=load(path);m=bc.layers[d['key']];s=bc.make(d,'C');c=contributions(d['q'],s,m.basis,d['log_scale']);mask,r,a=route(d['q'][qs],s,m,d['own']);event=str(path.relative_to(OUT));panel=np.load(E25/f'C_s{seed}'/f'panel_{idx:04d}.npz');assert event==str(panel['event']);assert np.array_equal(mask.cpu(),panel['states'][0]);assert np.array_equal(r.cpu(),panel['initial_remove']) and np.array_equal(a.cpu(),panel['initial_add'])
  tz,tu,tb,t=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ']);assert torch.isfinite(tz).all() and torch.isfinite(tb).all();lm=c['logzh']-c['A'].log()[:,None]
  F=features(bc.reader,d['q'],c);Fd=features(bd.reader,d['q'],c)
  zc,betac,cc,ac=fit_coeff(F,lm,tz,tb,fit['zscale'],fit['bscale']);zd,betad,cd,ad=fit_coeff(Fd,lm,tz,tb,fit['zscale'],fit['bscale']);zs,sv=svd_rank5(tz)
  models={'OracleCoeff_Cphi':(zc,betac),'OracleCoeff_Dphi':(zd,betad),'Cphi_trueBeta':(zc,tb),'SVD5_trueBeta':(zs,tb)}
  J=float(objective(c,d['N'],d['Z'],d['dense'],mask));g=torch.as_tensor(panel['true_initial_gain'],device='cuda');oracle=max(0.,float(g.max()));normalizer=d['dense'][qs].square().sum()/16+1e-12
  state_masks=torch.as_tensor(panel['states'],device='cuda');rr=torch.as_tensor(panel['remove'],device='cuda');aa=torch.as_tensor(panel['add'],device='cuda');ts=panel['true_gain'];outpanels={};init_scores={}
  np.savez_compressed(dest/f'coeff_{idx:04d}.npz',event=np.array(event),**{'C_'+k:v.cpu().numpy() for k,v in cc.items()},**{'D_'+k:v.cpu().numpy() for k,v in cd.items()},singular_values_scaled=sv.cpu().numpy())
  for name,(zz,bb) in models.items():
   frow=dict(event=event,seed=seed,method=name,z_nmse=nmse(zz,tz),beta_nmse=nmse(bb,tb),z_unselected_nmse=nmse(zz[:,~mask],tz[:,~mask]),max_abs_beta=float(bb.abs().max()))
   if name=='OracleCoeff_Cphi':frow.update(ac)
   if name=='OracleCoeff_Dphi':frow.update(ad)
   fits.append(frow)
   try:
    score,ind=safe_predictions(zz[qs],bb[qs],lm[qs],mask,r,a,normalizer);ps=[]
    for st in state_masks:ps.append(safe_predictions(zz[qs],bb[qs],lm[qs],st,rr,aa,normalizer)[0].cpu().numpy())
    ps=np.stack(ps);outpanels[name]=ps;init_scores[name]=score.cpu().numpy();init_scores[name+'_ind']=ind.cpu().numpy();conds.append(dict(event=event,seed=seed,method=name,initial=J,common_actions=len(rr),**conditional_metrics(ts,ps)))
    for policy,sc in [('Set',score),('Independent',ind)]:
     j=int(sc.argmax());ex=bool(sc[j]>0);true=float(g[j]);rows.append(dict(event=event,seed=seed,trajectory=d['ev']['trajectory'],layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head'],method=name,policy=policy,initial=J,oracle_gain=oracle,proposal_gain=true,hindsight_gain=max(0.,true),executed_gain=true if ex else 0.,execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(sc[j])))
   except FloatingPointError as exc:
    failures.append(dict(event=event,seed=seed,method=name,error=str(exc)));print('NUMERIC_UNRESOLVED',failures[-1],flush=True)
  # Reproduce learned references without touching their model, fit or states.
  cp={n:x[qs] for n,x in c.items()}
  for name,bundle in [('E24_C',bc),('E25_D',bd)]:
   zz,uu,bb=bundle.reader(d['q'][qs],s,cp);sc=predicted(zz,uu,cp['mass'],mask,r,a,bb)/normalizer
   allD=(cp['mass']*bb.double().exp()).sum(1);net=zz-zz.sum(1)[:,None]/allD[:,None,None]*uu[...,None];energy=(net.square().sum(-1)/allD[:,None].square()).mean(0);ind=energy[a]-energy[r]
   for policy,score in [('Set',sc),('Independent',ind)]:
    j=int(score.argmax());ex=bool(score[j]>0);true=float(g[j]);rows.append(dict(event=event,seed=seed,trajectory=d['ev']['trajectory'],layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head'],method=name,policy=policy,initial=J,oracle_gain=oracle,proposal_gain=true,hindsight_gain=max(0.,true),executed_gain=true if ex else 0.,execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(score[j])))
   safe,_=safe_predictions(zz,bb,lm[qs],mask,r,a,normalizer);err=float((safe-sc).abs().max());audits.append(dict(event=event,model=name,safe_incremental_max_error=err))
   saved=np.load(E25/f'{"C" if name=="E24_C" else "D"}_s{seed}'/f'panel_{idx:04d}.npz');assert int(sc.argmax())==int(saved['P0'].argmax()) and bool(sc.max()>0)==bool(saved['P0'].max()>0)
   outpanels[name]=saved['predicted_gain'];conds.append(dict(event=event,seed=seed,method=name,initial=J,common_actions=len(rr),**conditional_metrics(ts,saved['predicted_gain'])))
  for name,pkey in [('P1','P1'),('P3','P3')]:
   sc=torch.as_tensor(panel[pkey],device='cuda');j=int(sc.argmax());ex=bool(sc[j]>0);true=float(g[j]);rows.append(dict(event=event,seed=seed,trajectory=d['ev']['trajectory'],layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head'],method=name,policy='Set',initial=J,oracle_gain=oracle,proposal_gain=true,hindsight_gain=max(0.,true),executed_gain=true if ex else 0.,execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(sc[j])))
  rows.append(dict(event=event,seed=seed,trajectory=d['ev']['trajectory'],layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head'],method='NoOp',policy='NoOp',initial=J,oracle_gain=oracle,proposal_gain=0.,hindsight_gain=0.,executed_gain=0.,execute=False,remove=-1,add=-1,predicted_gain=0.))
  np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=np.array(event),states=panel['states'],remove=panel['remove'],add=panel['add'],true_gain=ts,**outpanels,**{'initial_'+k:v for k,v in init_scores.items()})
  if idx%24==0:dump(dest/'status.json',dict(done=idx,total=len(paths),seconds=time.time()-start));print(seed,idx,time.time()-start,flush=True)
 dump(dest/'decisions.json',rows);dump(dest/'conditional.json',conds);dump(dest/'fit.json',fits);dump(dest/'numerical_failures.json',failures);dump(dest/'audit.json',audits);dump(dest/'complete.json',dict(seed=seed,events=len(indices),seconds=time.time()-start,C_sha=sha(checkpoint('C',seed)),D_sha=sha(checkpoint('D',seed)),fit_sha=sha(OUT/'fit.pt'),ridge=RIDGE,failures=len(failures),test_accessed=False,training=False));print('DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--limit',type=int);a=p.parse_args();run(a.seed,a.limit)
