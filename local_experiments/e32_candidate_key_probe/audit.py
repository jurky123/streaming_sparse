from .run import *

@torch.no_grad()
def main():
 torch.set_num_threads(2);paths=files('development');manifest=json.loads((DEST/'plans/manifest.json').read_text());cache={};plans=[];checks=[];counts=[];probes=[];qs=torch.linspace(0,63,16).round().long()
 for idx,path in enumerate(paths):
  ev=torch.load(path,map_location='cpu',weights_only=True);kp=path.parent/ev['kv_file']
  if kp not in cache:cache[kp]=torch.load(kp,map_location='cpu',weights_only=True)['ids']
  ids=cache[kp];sp=np.load(DEST/'plans'/f'{idx:04d}.npz');assert sha(DEST/'plans'/f'{idx:04d}.npz')==manifest['events'][idx]['plan_sha'];assert digest(ids)==str(sp['ids_sha'])
  for rs in SAMPLE_SEEDS:
   perm=sp[f's{rs}'];assert np.array_equal(perm,permutations(ids,ev['trajectory'],ev['layer'],ev['head'],rs));assert np.all(np.sort(perm,-1)==np.arange(16))
  plans.append(dict(event=str(sp['event']),ids_sha=str(sp['ids_sha']),metadata_only_sampling_reproduced=True,nested=True))
 for seed in [17,29]:
  dest=DEST/f's{seed}';rows=json.loads((dest/'decisions.json').read_text());bundle=bundle_for('C',seed);panels={};eventidx={}
  for idx in range(288):
   p=np.load(dest/f'panel_{idx:04d}.npz');panels[str(p['event'])]=p;eventidx[str(p['event'])]=idx
  for row in rows:
   p=panels[row['event']];mask=p['states'][0];own=len(mask)-72+row['tile'];r,a=row['remove'],row['add'];assert mask[r] and not mask[a] and r!=own and mask[own];assert np.any((p['initial_remove']==r)&(p['initial_add']==a));new=mask.copy()
   if row['execute']:new[r]=False;new[a]=True
   assert new.sum()==mask.sum()==max(1,int(len(mask)*.1)) and new[own]
  counts.append(dict(seed=seed,decisions=len(rows),all_count_own_panel_valid=True))
  worst=min([r for r in rows if r['policy']=='Set' and r['initial']>1e-12],key=lambda r:r['executed_gain']/r['initial'])['event']
  for idx in sorted(set([0,47,143,144,230,eventidx[worst]])):
   d,s,c,_,_,_,_=prepare_local(bundle,paths[idx]);ev=d['ev'];p=panels[str(paths[idx].relative_to(OUT))];x=torch.load(E30/f's{seed}'/f'beta_{idx:04d}.pt',map_location='cpu',weights_only=True);sp=np.load(DEST/'plans'/f'{idx:04d}.npz');cand=p['candidate_pages'];q=d['q'][qs].cpu().numpy().astype(np.longdouble);kg=d['k'][0].double().cpu().numpy().astype(np.longdouble).reshape(-1,4,16,128);sk=s.keys[0].double().cpu().numpy().astype(np.longdouble)
   def lse(x,axis=-1):
    mx=x.max(axis,keepdims=True);return (mx+np.log(np.exp(x-mx).sum(axis,keepdims=True))).squeeze(axis)
   logzh=lse(np.einsum('qd,pgd->qpg',q,sk[cand])/np.sqrt(np.longdouble(128)),axis=-1)+np.log(np.longdouble(16));assert np.max(np.abs(logzh-p['summary_logz_candidate']))<1e-8
   for rs in SAMPLE_SEEDS:
    per=sp[f's{rs}'][cand]
    for b in BS:
     kk=np.take_along_axis(kg[cand],per[:,:,:b,None].astype(int),axis=2);lg=np.einsum('qd,pgtd->qpgt',q,kk)/np.sqrt(np.longdouble(128));estimate=lse(lg.reshape(16,48,-1))+np.log(np.longdouble(16)/b)-logzh;err=float(np.max(np.abs(estimate-p[f'beta_KeyProbe-b{b}-r{rs}'])));assert err<1e-8;probes.append(dict(seed=seed,event=x['event'],sampling_seed=rs,b=b,independent_probe_max_error=err))
   # b=16 is ONLY an all-keys numerical identity audit, never an evaluated probing configuration.
   lg=np.einsum('qd,pgtd->qpgt',q,kg[cand])/np.sqrt(np.longdouble(128));full=lse(lg.reshape(16,48,64))-logzh;assert np.max(np.abs(full-p['true_beta_candidate']))<1e-7
   N=d['N'].cpu().numpy().astype(np.longdouble);Z=d['Z'].cpu().numpy().astype(np.longdouble);NH=c['nh'].cpu().numpy().astype(np.longdouble);ZH=c['zh'].cpu().numpy().astype(np.longdouble);OD=d['dense'].cpu().numpy().astype(np.longdouble)
   def objective_np(mask):
    o=np.where(mask[None,:,None],N,NH).sum(1)/np.where(mask[None],Z,ZH).sum(1)[:,None];return ((OD-o)**2).sum()/((OD**2).sum()+np.longdouble(1e-12))
   mask=p['states'][0];J=objective_np(mask);norm=float(d['dense'][qs].square().sum()/16+1e-12);l=x['logmass'][qs].numpy().astype(np.longdouble);cen=x['center'][qs].numpy().astype(np.longdouble);v=cen+x['predicted_response'][qs].numpy().astype(np.longdouble)
   def softmax(z):
    w=np.exp(z-z.max(1,keepdims=True));return w/w.sum(1,keepdims=True)
   for row in [r for r in rows if r['event']==x['event']]:
    method=row['method'];b=x['betas']['Predicted'][qs].numpy().astype(np.longdouble).copy()
    if method=='Full-True':b=x['beta_true'][qs].numpy().astype(np.longdouble)
    elif method=='Candidate-True':b[:,cand]=x['beta_true'][qs].numpy()[:,cand]
    elif method.startswith('Key'):b[:,cand]=p[f"beta_{method}-r{row['sampling_seed']}"]
    target=(softmax(l+b)[...,None]*v).sum(1)
    def hybrid(m):return (softmax(np.where(m[None],l+b,l))[...,None]*np.where(m[None,:,None],v,cen)).sum(1)
    new=mask.copy();new[row['remove']]=False;new[row['add']]=True;g=J-objective_np(new);err=abs(float(g)-row['proposal_gain']);assert err<1e-8
    perr=None
    if row['policy']=='Set':
     H=hybrid(mask);delta=hybrid(new)-H;raw=(2*(target-H)*delta-delta*delta).sum(-1).mean();pred=raw/norm;perr=abs(float(pred)-row['predicted_gain']);assert perr<1e-8+1e-6*abs(float(pred));assert (raw>0)==row['execute'] or abs(float(raw))<1e-18
    checks.append(dict(seed=seed,event=x['event'],method=method,sampling_seed=row['sampling_seed'],policy=row['policy'],true_gain_error=err,predicted_score_error=perr,positive_mixture_finite=bool(np.isfinite(target).all() and np.isfinite(hybrid(new)).all())))
  done=json.loads((dest/'complete.json').read_text());assert done['reader_sha']==sha(E29/f's{seed}/selected.pt') and done['C_sha']==sha(checkpoint('C',seed)) and done['phi_sha']==sha(E27/f's{seed}/selected.pt')
 dump(DEST/'independent_audit.json',dict(counts=counts,plans=plans,checks=checks,probe_checks=probes,all_keys_identity_audit_only=True,all_frozen_hashes_unchanged=True,training=False,test_accessed=False));print('AUDIT PASSED',sum(r['decisions'] for r in counts),len(checks),len(probes),flush=True)
if __name__=='__main__':main()
