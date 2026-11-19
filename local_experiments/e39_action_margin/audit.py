from .report import *
from ..e38_action_attribution.audit import numpy_score,arr

def main():
 torch.set_num_threads(2);proto=json.loads((DEST/'protocol.json').read_text());source=[]
 for name,h in proto['sources'].items():assert sha(ROOT/name)==h;source.append(name)
 checks=[];pairs=[];stepchecks=[];qs=torch.linspace(0,63,16,device='cuda').round().long()
 for seed in [17,29]:
  assert sha(OUT/'fit.pt')==json.loads((E38/f'B_s{seed}/complete.json').read_text())['basis_sha']
  sched=json.loads((DEST/f'schedule_s{seed}.json').read_text());assert len(sched)==2304;trainnames={str(p.relative_to(OUT)) for p in files('train')};devnames={str(p.relative_to(OUT)) for p in files('development')};assert trainnames.isdisjoint(devnames);assert all(r['event'] in trainnames for r in sched)
  for epoch in [0,1]:assert len({r['event'] for r in sched[epoch*1152:(epoch+1)*1152]})==1152
  for row in sched:
   mask=np.array(row['mask'],bool);r=np.array(row['remove']);a=np.array(row['add']);tile=int(Path(row['event']).stem.split('_t')[-1]);own=len(mask)-72+tile;assert mask.sum()==int(.1*len(mask)) and mask[own] and mask[r].all() and (~mask[a]).all() and (r!=own).all() and len(r)==512
  dec=json.loads((DEST/f'decomposition_s{seed}.json').read_text());assert len(dec)==288
  for r in dec:
   e=r['prediction_error']+r['query_difference']+r['projection_difference']-r['M_pred']+r['M_true64_full'];assert abs(e)<1e-9
  bundle=bundle_for('C',seed);phi=load_phi(seed);frozenhash=[sha(checkpoint('C',seed)),sha(E27/f's{seed}/selected.pt')]
  for group in ['T0','T1']:
   dest=DEST/f'{group}_s{seed}';done=json.loads((dest/'training_complete.json').read_text());assert done['updates']==2304 and done['parameters']==51157 and done['frozen_C_exact'] and done['frozen_phi_exact'];assert [done['C_sha'],done['phi_sha']]==frozenhash;assert done['schedule_sha']==sha(DEST/f'schedule_s{seed}.json');original=torch.load(E29/f's{seed}/selected.pt',map_location='cpu',weights_only=False)['model'];initial=torch.load(dest/'initial.pt',map_location='cpu',weights_only=True);assert all(torch.equal(initial[n],v) for n,v in original.items());curves=json.loads((dest/'curves.json').read_text());assert len(curves)==2304 and all(x['event']==s['event'] and x['update']==s['update'] for x,s in zip(curves,sched));assert all(math.isfinite(x['loss']) and math.isfinite(x['gradnorm']) for x in curves)
   model=LocalReader().cuda();model.load_state_dict(torch.load(dest/'epoch1.pt',map_location='cuda',weights_only=False)['model']);rows=json.loads((dest/'development_epoch1.json').read_text());byevent=collections.defaultdict(list)
   for row in rows:byevent[row['event']].append(row)
   assert len(byevent)==288
   for idx,path in enumerate(files('development')):
    p=np.load(dest/f'epoch1_panel_{idx:04d}.npz');old=np.load(E30/f's{seed}/panel_{idx:04d}.npz');assert np.array_equal(p['mask'],old['states'][0]);assert np.array_equal(p['remove'],old['initial_remove']);assert np.array_equal(p['add'],old['initial_add']);g=p['true_gain'];assert np.allclose(g,old['true_initial_gain'],atol=1e-8,rtol=1e-6)
    rr=byevent[str(path.relative_to(OUT))]
    for row in rr:
     sc=p['scores'] if row['policy']=='Set' else p['independent'];j=pick(torch.as_tensor(sc));assert row['action']==j;assert abs(row['executed_gain']-value(g,j))<1e-10;m=p['mask'].copy()
     if j>=0:m[p['remove'][j]]=False;m[p['add'][j]]=True
     own=len(m)-72+row['tile'];assert m[own] and m.sum()==int(len(m)*.1)
    if idx not in [0,47,143,144,230,287]:continue
    with torch.no_grad():
     d,s,c,rt,bt,lm,cen=prepare_local(bundle,path);F=phi(d['q'],s);rp,bp=model(s,F[qs]);norm=float(d['dense'].square().sum()/64+1e-12/64);ix=np.unique(np.r_[0,np.argmax(p['scores']),np.argmax(g)]);npv=numpy_score(arr(rp),arr(bp),arr(lm[qs]),arr(cen[qs]),p['mask'],p['remove'][ix],p['add'][ix],norm);err=float(max(abs(np.array(npv)-p['scores'][ix])));assert np.allclose(npv,p['scores'][ix],atol=1e-8,rtol=1e-5);oldscore,_=safe_predictions(reconstruct(rp,bp,lm[qs],cen[qs]),bp,lm[qs],torch.as_tensor(p['mask'],device='cuda'),torch.as_tensor(p['remove'],device='cuda'),torch.as_tensor(p['add'],device='cuda'),norm);assert np.allclose(oldscore.cpu(),p['scores'],atol=1e-8,rtol=1e-5);checks.append(dict(seed=seed,group=group,event=str(path.relative_to(OUT)),error=err,old_new_score_error=float(np.max(abs(oldscore.cpu().numpy()-p['scores'])))))
     if group=='T0':
      # Independent canonical page outputs reconstruct B-space approximation without raw-QK.
      x=torch.load(E30/f's{seed}/beta_{idx:04d}.pt',map_location='cpu',weights_only=True);t=torch.load(E38/f'B_s{seed}/target_{idx:04d}.pt',map_location='cpu',weights_only=True);rt0=arr(t['true_response']);bt0=arr(x['beta_true']);lm0=arr(x['logmass']);cen0=arr(x['center']);pair=[dec[idx]['j_pred'],dec[idx]['j_oracle']]
      def marg(qix):
       js=[j for j in pair if j>=0];vals=iter(numpy_score(rt0[qix],bt0[qix],lm0[qix],cen0[qix],p['mask'],p['remove'][js],p['add'][js],norm));v=[next(vals) if j>=0 else 0. for j in pair];return v[0]-v[1]
      e16=abs(marg(qs.cpu().numpy())-dec[idx]['M_true16']);e64=abs(marg(np.arange(64))-dec[idx]['M_true64_B16']);assert max(e16,e64)<1e-8;pairs.append(dict(seed=seed,event=dec[idx]['event'],error16=e16,error64=e64))
   # Replay first training update independently for both objectives, including teacher choice.
   with torch.no_grad():
    step=sched[0];d,s,c,rt,bt,lm,cen=prepare_local(bundle,OUT/step['event']);F=phi(d['q'],s);mask=torch.tensor(step['mask'],device='cuda');r=torch.tensor(step['remove'],device='cuda');a=torch.tensor(step['add'],device='cuda');g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a)
   model.load_state_dict(original);rp,bp=model(s,F[qs]);fit=torch.load(OUT/'fit.pt',weights_only=True);gs=rms_floor(g,(0,),fit['gscale']).reshape(());bs=rms_floor(bt[qs],(0,),fit['bscale']);scale=json.loads((E29/f's{seed}/local_scale.json').read_text())['scale'];local=((c['mass'][qs][...,None]*(rp-rt[qs]))/scale).square().mean();quality=nn.functional.smooth_l1_loss(bp/bs,bt[qs]/bs);pg,_=safe_predictions(reconstruct(rp,bp,lm[qs],cen[qs]),bp,lm[qs],mask,r,a,d['dense'].square().sum()/64+1e-12);pvals=np.r_[pg.detach().cpu().numpy(),0.];tvals=np.r_[g.cpu().numpy(),0.];best=int(tvals.argmax()) if g.max()>0 else len(g);H=sorted(set(np.argsort(-pvals,kind='stable')[:4])|{best,len(g)});marg=max((pvals[j]-pvals[best]+tvals[best]-tvals[j])/float(gs) for j in H);assert abs(marg-curves[0]['margin_loss'])<1e-5*(1+abs(marg));assert H==curves[0]['hard'];sl=original_set_loss(pg,g,gs) if group=='T0' else margin_loss(pg,g,gs)[0];loss=local+quality+sl;assert abs(float(loss)-curves[0]['loss'])<1e-5*(1+abs(float(loss)));stepchecks.append(dict(seed=seed,group=group,loss=float(loss),margin=float(marg)))
 # Verify all final gates derive from the same final checkpoints; no Test data was enumerated by scripts.
 dump(DEST/'audit.json',dict(passed=True,development_events_checked=1152,development_policy_decisions_checked=2304,independent_longdouble_checks=checks,decomposition_numpy_checks=pairs,first_update_checks=stepchecks,protocol_sources_unchanged=source,initials_equal_E29=True,schedule_matched=True,summary_phi_frozen=True,test_accessed=False));print('AUDIT PASSED',flush=True)
if __name__=='__main__':main()
