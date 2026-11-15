from .core import *

@torch.no_grad()
def summary_only_inputs(q,s,basis,B):
 # No raw KV, teacher N/Z, Dense output, or teacher log-scale argument.
 logits=torch.einsum('qd,pgd->qpg',q.double(),s.keys[0].double())/math.sqrt(128)
 lz=(logits+math.log(16)).logsumexp(-1);lm=lz-lz.logsumexp(-1)[:,None];ap=summary_output({'logit':logits},s,basis);o=(lm.exp()[...,None]*ap).sum(1)
 return lm,(ap-o[:,None])@B

@torch.no_grad()
def main():
 torch.set_num_threads(2);checks=[];counts=[];B=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda().double();qs=torch.linspace(0,63,16,device='cuda').round().long()
 for seed in [17,29]:
  dest=E29/f's{seed}';done=json.loads((dest/'training_complete.json').read_text());model=LocalReader().cuda();model.load_state_dict(torch.load(dest/'selected.pt',map_location='cuda',weights_only=False)['model']);phi=load_phi(seed);bundle=bundle_for('C',seed);rows=json.loads((dest/f"development_epoch{done['best']['epoch']}.json").read_text());panels={}
  for p in (E25/f'C_s{seed}').glob('panel_*.npz'):
   n=np.load(p);panels[str(n['event'])]=dict(mask=n['states'][0],r=n['initial_remove'],a=n['initial_add'])
  for row in rows:
   p=panels[row['event']];mask=p['mask'];own=len(mask)-72+row['tile'];rr,aa=row['remove'],row['add'];assert mask[rr] and not mask[aa] and rr!=own and mask[own];assert np.any((p['r']==rr)&(p['a']==aa));m=mask.copy()
   if row['execute']:m[rr]=False;m[aa]=True
   assert m.sum()==mask.sum()==max(1,int(len(m)*.1)) and m[own]
  counts.append(dict(seed=seed,decisions=len(rows),all_count_own_panel_valid=True))
  valid=[x for x in rows if x['method']=='Local-Online' and x['policy']=='Set' and x['initial']>1e-12]
  worst=min(valid,key=lambda x:x['executed_gain']/x['initial'])['event'];index={str(p.relative_to(OUT)):i for i,p in enumerate(files('development'))}
  for idx in sorted(set([0,47,143,144,230,index[worst]])):
   path=files('development')[idx];d,s,c,rt,bt,lm,center=prepare_local(bundle,path);F=phi(d['q'],s);rp,bp=model(s,F[qs]);mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);z=reconstruct(rp,bp,lm[qs],center[qs]);score,_=safe_predictions(z,bp,lm[qs],mask,r,a,1.);j=int(score.argmax());rawfree_lm,rawfree_center=summary_only_inputs(d['q'][qs],s,bundle.layers[d['key']].basis,B);zf=reconstruct(rp,bp,rawfree_lm,rawfree_center);sf,_=safe_predictions(zf,bp,rawfree_lm,mask,r,a,1.);jj=int(sf.argmax());diff=float((score-sf).abs().max());assert j==jj or diff<1e-10;assert bool(score[j]>0)==bool(sf[jj]>0)
   saved=next(x for x in rows if x['event']==str(path.relative_to(OUT)) and x['method']=='Local-Online' and x['policy']=='Set');assert (saved['remove'],saved['add'])==(int(r[j]),int(a[j]));assert saved['execute']==bool(score[j]>0)
   # longdouble mixture reference, independent of reconstructed residual formula.
   zz=z.cpu().numpy().astype(np.longdouble);mm=np.exp(lm[qs].cpu().numpy().astype(np.longdouble));ee=np.exp((lm[qs]+bp.double()).cpu().numpy().astype(np.longdouble));m=mask.cpu().numpy();new=m.copy();new[int(r[j])]=False;new[int(a[j])]=True;target=zz.sum(1)/ee.sum(1)[:,None];h=zz[:,m].sum(1)/np.where(m,ee,mm).sum(1)[:,None];h2=zz[:,new].sum(1)/np.where(new,ee,mm).sum(1)[:,None];delta=h2-h;ref=(2*(target-h)*delta-delta*delta).sum(-1).mean();err=abs(float(ref)-float(score[j]));assert err<1e-8+1e-6*abs(float(ref))
   N=d['N'].cpu().numpy().astype(np.longdouble);Z=d['Z'].cpu().numpy().astype(np.longdouble);NH=c['nh'].cpu().numpy().astype(np.longdouble);ZH=c['zh'].cpu().numpy().astype(np.longdouble);OD=d['dense'].cpu().numpy().astype(np.longdouble)
   def objective_np(x):
    o=np.where(x[None,:,None],N,NH).sum(1)/np.where(x[None],Z,ZH).sum(1)[:,None];return ((OD-o)**2).sum()/((OD**2).sum()+np.longdouble(1e-12))
   gt=objective_np(m)-objective_np(new);gerr=abs(float(gt)-saved['proposal_gain']);assert gerr<1e-8
   checks.append(dict(seed=seed,event=str(path.relative_to(OUT)),summary_only_same_action=j==jj,summary_only_score_max_difference=diff,predicted_score_reference_error=err,true_gain_error=gerr,beta_min=float(bp.min()),beta_max=float(bp.max()),predicted_denominators_positive=True))
 dump(E29/'independent_audit.json',dict(checks=checks,counts=counts,test_accessed=False));print('AUDIT DONE',flush=True)
if __name__=='__main__':main()
