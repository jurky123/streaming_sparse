from .run import *
from .report import table

@torch.no_grad()
def main():
 torch.set_num_threads(2);paths=files('development');checks=[];counts=[];hashes={};stages=['A']+(['B'] if all((DEST/f'B_s{s}/complete.json').exists() for s in [17,29]) else [])
 def record(path):hashes[str(path.relative_to(ROOT))]=sha(path)
 def ar(t):return t.cpu().double().numpy().astype(np.longdouble)
 def softmax(a):
  w=np.exp(a-a.max(1,keepdims=True));return w/w.sum(1,keepdims=True)
 for seed in [17,29]:
  bundle=bundle_for('C',seed);record(checkpoint('C',seed));record(E29/f's{seed}/selected.pt')
  for stage in stages:
   dest=DEST/f'{stage}_s{seed}';rows=json.loads((dest/'decisions.json').read_text());panels={};eventidx={}
   for i in range(288):
    pp=dest/f'panel_{i:04d}.npz';p=np.load(pp);old=np.load(E30/f's{seed}/panel_{i:04d}.npz');event=str(p['event']);panels[event]=p;eventidx[event]=i
    for key in ['states','remove','add','initial_remove','initial_add','true_initial_gain','true_gain']:assert np.array_equal(p[key],old[key])
    assert event==str(paths[i].relative_to(OUT));record(pp);record(E30/f's{seed}/panel_{i:04d}.npz');record(E30/f's{seed}/beta_{i:04d}.pt')
   for row in rows:
    p=panels[row['event']];m=p['states'][0];own=len(m)-72+row['tile'];r,a=row['remove'],row['add'];assert m[r] and not m[a] and r!=own and m[own];assert np.any((p['initial_remove']==r)&(p['initial_add']==a));new=m.copy()
    if row['execute']:new[r]=False;new[a]=True
    assert new.sum()==m.sum()==int(.1*len(m)) and new[own]
   counts.append(dict(seed=seed,stage=stage,decisions=len(rows),events=288,original_masks_pools_true_labels_exact=True,count_own_panel_valid=True))
   worst=max([r for r in rows if r['policy']=='Set'],key=lambda r:r['absolute_nmse_increase'])['event']
   for idx in sorted(set([0,47,143,144,230,eventidx[worst]])):
    d,s,c,*_=prepare_local(bundle,paths[idx]);event=str(paths[idx].relative_to(OUT));p=panels[event];x=torch.load(E30/f's{seed}/beta_{idx:04d}.pt',weights_only=True);N,Z,NH,ZH,OD=map(ar,[d['N'],d['Z'],c['nh'],c['zh'],d['dense']]);m=p['states'][0]
    def obj(mm,qs=np.arange(64),w=None):
     out=np.where(mm[None,:,None],N,NH).sum(1)/np.where(mm[None],Z,ZH).sum(1)[:,None];er=((out-OD)**2).sum(-1);en=(OD**2).sum(-1)
     if w is None:return er[qs].sum()/(en[qs].sum()+np.longdouble(1e-12))
     return (w*er[qs]).sum()/((w*en[qs]).sum()+np.longdouble(1e-12/64))
    J=obj(m)
    if stage=='B':
     q=ar(d['q']);selected=[int(np.argmin(((q-q.mean(0))**2).sum(-1)))];dist=np.full(64,np.inf,dtype=np.longdouble)
     for k in range(16):
      if k:dist[selected]=-np.inf;selected.append(int(np.argmax(dist)))
      dist=np.minimum(dist,((q-q[selected[-1]])**2).sum(-1))
     DD=((q[:,None]-q[selected][None])**2).sum(-1);assignment=DD.argmin(1);w=np.bincount(assignment,minlength=16)/64;assert np.array_equal(selected,p['ids']) and np.array_equal(assignment,p['assignment']) and np.array_equal(w,p['weights'])
    for row in [r for r in rows if r['event']==event]:
     qs=p['ids'] if stage=='B' else np.linspace(0,63,16).round().astype(int) if row['method'].endswith('M16') else np.arange(64);w=p['weights'] if stage=='B' else np.full(len(qs),1/len(qs));new=m.copy();new[row['remove']]=False;new[row['add']]=True;gain=J-obj(new);err=abs(float(gain)-row['proposal_gain']);assert err<1e-8;perr=None
     if row['policy']=='Set':
      if row['method'].startswith('True-'):score=obj(m,qs,w if stage=='B' else None)-obj(new,qs,w if stage=='B' else None)
      else:
       lm=ar(x['logmass'])[qs];cen=ar(x['center'])[qs];v=cen+ar(x['predicted_response'])[qs];beta=ar(x['beta_true'] if row['method'].startswith('TrueBeta') else x['betas']['Predicted'])[qs];target=(softmax(lm+beta)[...,None]*v).sum(1)
       def hybrid(mm):return (softmax(np.where(mm[None],lm+beta,lm))[...,None]*np.where(mm[None,:,None],v,cen)).sum(1)
       H=hybrid(m);delta=hybrid(new)-H;score=(w*(2*(target-H)*delta-delta*delta).sum(-1)).sum()/((w*(OD[qs]**2).sum(-1)).sum()+np.longdouble(1e-12))
      perr=abs(float(score)-row['predicted_gain']);assert perr<1e-8+1e-6*abs(float(score));assert (score>0)==row['execute'] or abs(float(score))<1e-18
     checks.append(dict(seed=seed,stage=stage,event=event,method=row['method'],policy=row['policy'],true_gain_error=err,predicted_gain_error=perr))
   done=json.loads((dest/'complete.json').read_text());assert done['C_sha']==sha(checkpoint('C',seed)) and done['reader_sha']==sha(E29/f's{seed}/selected.pt') and done['protocol_sha']==sha(DEST/'protocol.json')
 for path in (ROOT/'local_experiments/e36_query_coverage').glob('*.py'):record(path)
 record(DEST/'protocol.json');table('source_hashes.csv',[dict(path=k,sha256=v) for k,v in sorted(hashes.items())]);dump(DEST/'audit.json',dict(counts=counts,independent_numpy_longdouble_checks=checks,max_true_gain_error=max(x['true_gain_error'] for x in checks),max_prediction_error=max(x['predicted_gain_error'] or 0 for x in checks),frozen_hashes_unchanged=True,test_accessed=False,training=False,new_qkv=False));print('AUDIT PASSED',len(checks))
if __name__=='__main__':main()
