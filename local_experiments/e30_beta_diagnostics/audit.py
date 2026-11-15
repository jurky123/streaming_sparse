"""Independent longdouble selected-output audit and NumPy ridge recomputation."""
from .run import *
from ..e29_local_reader.audit import summary_only_inputs

@torch.no_grad()
def main():
 torch.set_num_threads(2);checks=[];counts=[];ls=[];B=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda().double();qs=torch.linspace(0,63,16).round().long();paths=files('development')
 for seed in [17,29]:
  dest=DEST/f's{seed}';rows=json.loads((dest/'decisions.json').read_text());bundle=bundle_for('C',seed);panels={};eventidx={}
  for i in range(288):
   n=np.load(dest/f'panel_{i:04d}.npz');panels[str(n['event'])]=n;eventidx[str(n['event'])]=i
  for row in rows:
   p=panels[row['event']];mask=p['states'][0];own=len(mask)-72+row['tile'];rr,aa=row['remove'],row['add'];assert mask[rr] and not mask[aa] and rr!=own and mask[own];assert np.any((p['initial_remove']==rr)&(p['initial_add']==aa));m=mask.copy()
   if row['execute']:m[rr]=False;m[aa]=True
   assert m.sum()==mask.sum()==max(1,int(len(m)*.1)) and m[own]
  counts.append(dict(seed=seed,decisions=len(rows),all_count_own_panel_valid=True))
  sub=[r for r in rows if r['method']=='Predicted' and r['policy']=='Set' and r['initial']>1e-12];worst=min(sub,key=lambda r:r['executed_gain']/r['initial'])['event']
  for idx in sorted(set([0,47,143,144,230,eventidx[worst]])):
   path=paths[idx];d,s,c,rt,bt,lm,center=prepare_local(bundle,path);n=panels[str(path.relative_to(OUT))];cache=torch.load(dest/f'beta_{idx:04d}.pt',weights_only=True);rawfree_lm,rawfree_center=summary_only_inputs(d['q'][qs],s,bundle.layers[d['key']].basis,B)
   assert torch.allclose(rawfree_lm.cpu(),cache['logmass'][qs],atol=1e-9,rtol=1e-9);assert torch.allclose(rawfree_center.cpu(),cache['center'][qs],atol=1e-8,rtol=1e-8)
   # Separate numpy normal equations; same relative ridge, no torch fit helper.
   for p in [0,len(n['states'][0])-1]:
    F=cache['F'][:,p].numpy();l=cache['logmass'][:,p].numpy();b=cache['beta_true'][:,p].numpy();w=np.exp(l-l.max());X=F*w[:,None];G=X.T@X;coef=np.linalg.solve(G+np.eye(5)*1e-6*np.trace(G)/5,X.T@(w*b));pred=F@coef;err=float(np.max(np.abs(pred-cache['betas']['OracleCoeff64'][:,p].numpy())));assert err<1e-6;ls.append(dict(seed=seed,event=cache['event'],page=p,independent_ls_max_error=err))
   N=d['N'].cpu().numpy().astype(np.longdouble);Z=d['Z'].cpu().numpy().astype(np.longdouble);NH=c['nh'].cpu().numpy().astype(np.longdouble);ZH=c['zh'].cpu().numpy().astype(np.longdouble);OD=d['dense'].cpu().numpy().astype(np.longdouble)
   def objective_np(mask):
    o=np.where(mask[None,:,None],N,NH).sum(1)/np.where(mask[None],Z,ZH).sum(1)[:,None];return ((OD-o)**2).sum()/((OD**2).sum()+np.longdouble(1e-12))
   m=n['states'][0];J=objective_np(m);normalizer=float(d['dense'][qs].square().sum()/16+1e-12)
   for method in METHODS:
    beta=cache['betas'][method][qs];rp=cache['predicted_response'][qs];ll=cache['logmass'][qs];cc=cache['center'][qs]
    # Independent longdouble positive mixture; no reconstructed z or common scale.
    b=beta.numpy().astype(np.longdouble);l=ll.numpy().astype(np.longdouble);v=cc.numpy().astype(np.longdouble)+rp.numpy().astype(np.longdouble);cen=cc.numpy().astype(np.longdouble)
    def softmax(x):
     w=np.exp(x-x.max(1,keepdims=True));return w/w.sum(1,keepdims=True)
    target=(softmax(l+b)[...,None]*v).sum(1)
    def hybrid(x):return (softmax(np.where(x[None],l+b,l))[...,None]*np.where(x[None,:,None],v,cen)).sum(1)
    H=hybrid(m)
    for policy in ['Set','Independent']:
     row=next(r for r in rows if r['event']==cache['event'] and r['method']==method and r['policy']==policy);new=m.copy();new[row['remove']]=False;new[row['add']]=True;gtrue=J-objective_np(new);gerr=abs(float(gtrue)-row['proposal_gain']);assert gerr<1e-8,(seed,idx,method,gerr)
     if policy=='Set':
      H2=hybrid(new);delta=H2-H;pred=(2*(target-H)*delta-delta*delta).sum(-1).mean()/normalizer;perr=abs(float(pred)-row['predicted_gain']);assert perr<1e-8+1e-6*abs(float(pred)),(seed,idx,method,perr,float(pred))
     else:perr=None
     checks.append(dict(seed=seed,event=cache['event'],method=method,policy=policy,true_gain_error=gerr,predicted_score_error=perr,denominators_positive=bool(np.isfinite(target).all() and np.isfinite(H).all() and np.isfinite(hybrid(new)).all())))
  done=json.loads((dest/'complete.json').read_text());assert done['reader_sha']==sha(E29/f's{seed}'/'selected.pt') and done['C_sha']==sha(checkpoint('C',seed)) and done['phi_sha']==sha(E27/f's{seed}'/'selected.pt')
 dump(DEST/'independent_audit.json',dict(counts=counts,checks=checks,ls=ls,training=False,test_accessed=False,all_frozen_hashes_unchanged=True));print('AUDIT PASSED',sum(r['decisions'] for r in counts),len(checks),flush=True)
if __name__=='__main__':main()
