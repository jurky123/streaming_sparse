from .run import *

@torch.no_grad()
def main():
 torch.set_num_threads(2);paths=files('development');train={str(p.relative_to(OUT)) for p in files('train')};checks=[];counts=[];bchecks=[];bases_audit=[];hashes={};qs=torch.linspace(0,63,16).round().long()
 for seed in [17,29]:
  dest=DEST/f's{seed}';lock=json.loads((dest/'basis_lock.json').read_text());basis=torch.load(dest/'basis.pt',weights_only=True);assert lock['basis_sha']==sha(dest/'basis.pt')
  for source in lock['sources']:
   assert source['event'] in train;path=OUT/source['kv']
   if str(path) not in hashes:hashes[str(path)]=sha(path)
   assert hashes[str(path)]==source['kv_sha']
  for a in lock['audit']:
   k=a['key'];U=basis['bases'][k].double();G=basis['grams'][k]/(a['unique_pages']*4);e=basis['spectra'][k][:8];orth=float((U.T@U-torch.eye(8)).abs().max());res=float((G@U-U*e).abs().max()/G.abs().max());assert orth<1e-6 and res<1e-6;bases_audit.append(dict(seed=seed,key=k,orthogonality_error=orth,relative_eigen_residual=res,train_only_sources=True))
  rows=json.loads((dest/'decisions.json').read_text());bundle=bundle_for('C',seed);panels={};eventidx={}
  for i in range(288):
   n=np.load(dest/f'panel_{i:04d}.npz');panels[str(n['event'])]=n;eventidx[str(n['event'])]=i
  for row in rows:
   p=panels[row['event']];m=p['states'][0];own=len(m)-72+row['tile'];r,a=row['remove'],row['add'];assert m[r] and not m[a] and r!=own and m[own];assert np.any((p['initial_remove']==r)&(p['initial_add']==a));new=m.copy()
   if row['execute']:new[r]=False;new[a]=True
   assert new.sum()==m.sum()==int(.1*len(m)) and new[own]
  counts.append(dict(seed=seed,decisions=len(rows),count_own_panel_valid=True))
  selected=[r for r in rows if r['method']=='LowRank8+Variance' and r['policy']=='Set'];worst=max(selected,key=lambda r:r['absolute_nmse_increase'])['event']
  for idx in sorted(set([0,47,143,144,230,eventidx[worst]])):
   d,s,c,_,_,_,_=prepare_local(bundle,paths[idx]);event=str(paths[idx].relative_to(OUT));p=panels[event];cache=torch.load(dest/f'beta_{idx:04d}.pt',weights_only=True);x=torch.load(E30/f's{seed}/beta_{idx:04d}.pt',weights_only=True);q=d['q'].cpu().numpy().astype(np.longdouble);raw=d['k'][0].double().cpu().numpy().astype(np.longdouble).reshape(-1,4,16,128);keys=s.keys[0].double().cpu().numpy().astype(np.longdouble);stats=s.stats[0].double().cpu().numpy().astype(np.longdouble);U=basis['bases'][d['key']].double().numpy().astype(np.longdouble);offset=raw.mean(2)-keys;co=cache['coefficients_bf16'].double().numpy().astype(np.longdouble)
   expected=torch.from_numpy((offset@U).astype(np.float64)).bfloat16();assert torch.equal(expected,cache['coefficients_bf16'])
   def lse(z):
    mx=z.max(-1,keepdims=True);return (mx+np.log(np.exp(z-mx).sum(-1,keepdims=True))).squeeze(-1)
   l=np.einsum('qd,pgd->qpg',q,keys)/np.sqrt(np.longdouble(128));mu=np.einsum('qd,pgd->qpg',q,offset)/np.sqrt(np.longdouble(128));lo=np.einsum('qr,pgr->qpg',q@U,co)/np.sqrt(np.longdouble(128));vb=np.einsum('qj,pgj->qpg',(q*q).reshape(64,8,16).sum(-1)/128,stats[:,:,:8]);rl=np.einsum('qd,pgtd->qpgt',q,raw)/np.sqrt(np.longdouble(128));vc=((rl-rl.mean(-1,keepdims=True))**2).mean(-1)
   vals={'Variance-only':lse(l+.5*vb)-lse(l),'Full-Mean+Variance':lse(l+mu+.5*vb)-lse(l),'LowRank8+Variance':lse(l+lo+.5*vb)-lse(l),'Exact-Q-Cumulant2':lse(l+mu+.5*vc)-lse(l),'TrueBeta':lse(rl.reshape(64,len(keys),64))-lse(l)-np.log(np.longdouble(16))}
   for name,val in vals.items():
    truth=cache['beta'][name].double().numpy().astype(np.longdouble);err=float(np.max(np.abs(val-truth)));relative=err/max(1,float(np.max(np.abs(val))));assert relative<1e-9;bchecks.append(dict(seed=seed,event=event,method=name,beta_absolute_error=err,beta_scaled_error=relative))
   online=online_estimates(d['q'],s.keys[0],s.stats[0],cache['coefficients_bf16'].cuda(),basis['bases'][d['key']].cuda())
   for k,name in enumerate(['Variance-only','LowRank8+Variance']):assert torch.allclose(online[k].cpu(),cache['beta'][name],atol=1e-9,rtol=1e-9)
   N=d['N'].cpu().numpy().astype(np.longdouble);Z=d['Z'].cpu().numpy().astype(np.longdouble);NH=c['nh'].cpu().numpy().astype(np.longdouble);ZH=c['zh'].cpu().numpy().astype(np.longdouble);OD=d['dense'].cpu().numpy().astype(np.longdouble)
   def obj(m):
    out=np.where(m[None,:,None],N,NH).sum(1)/np.where(m[None],Z,ZH).sum(1)[:,None];return ((out-OD)**2).sum()/((OD**2).sum()+np.longdouble(1e-12))
   m=p['states'][0];J=obj(m);norm=float(d['dense'][qs].square().sum()/16+1e-12);lm=x['logmass'][qs].numpy().astype(np.longdouble);cen=x['center'][qs].numpy().astype(np.longdouble);v=cen+x['predicted_response'][qs].numpy().astype(np.longdouble)
   def softmax(a):
    w=np.exp(a-a.max(1,keepdims=True));return w/w.sum(1,keepdims=True)
   for row in [r for r in rows if r['event']==event]:
    beta=cache['beta'][row['method']][qs].double().numpy().astype(np.longdouble);target=(softmax(lm+beta)[...,None]*v).sum(1)
    def hybrid(mm):return (softmax(np.where(mm[None],lm+beta,lm))[...,None]*np.where(mm[None,:,None],v,cen)).sum(1)
    new=m.copy();new[row['remove']]=False;new[row['add']]=True;gain=J-obj(new);err=abs(float(gain)-row['proposal_gain']);assert err<1e-8;perr=None
    if row['policy']=='Set':
     H=hybrid(m);delta=hybrid(new)-H;rawscore=(2*(target-H)*delta-delta*delta).sum(-1).mean();pred=rawscore/norm;perr=abs(float(pred)-row['predicted_gain']);assert perr<1e-8+1e-6*abs(float(pred));assert (rawscore>0)==row['execute'] or abs(float(rawscore))<1e-18
    checks.append(dict(seed=seed,event=event,method=row['method'],policy=row['policy'],true_gain_error=err,predicted_gain_error=perr,positive_mixture_finite=bool(np.isfinite(target).all() and np.isfinite(hybrid(new)).all())))
  repeat=json.loads((dest/'reuse.json').read_text());seen={}
  for r in repeat:
   ev=r['event'];group=re.search(r'_l(\d+)_s\d+_h(\d+)_',ev).groups();key=(str(Path(ev).parent),group,r['raw_sha'],r['summary_sha']);prior=seen.get(key)
   if prior is not None:assert prior==r['coef_and_variance_sha']
   seen[key]=r['coef_and_variance_sha']
  done=json.loads((dest/'complete.json').read_text());assert done['C_sha']==sha(checkpoint('C',seed)) and done['reader_sha']==sha(E29/f's{seed}/selected.pt') and done['basis_sha']==sha(dest/'basis.pt') and done['protocol_sha']==sha(DEST/'protocol.json')
 dump(DEST/'independent_audit.json',dict(counts=counts,checks=checks,beta_checks=bchecks,basis_checks=bases_audit,train_kv_files_verified=len(hashes),all_frozen_hashes_unchanged=True,all_reused_coefficients_identical=True,network_training=False,test_accessed=False));print('AUDIT PASSED',sum(r['decisions'] for r in counts),len(checks),len(bchecks),flush=True)
if __name__=='__main__':main()
