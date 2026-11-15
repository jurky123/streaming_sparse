from .core import *
MODES=('full_change','interaction_only')
def ids(sp):return sorted(p.name for p in (OUT/'events').iterdir() if split(p.name)==sp)
def deltas(d,mode):
 n=len(d['gains']);i,j=np.triu_indices(n,1);X=d['phi'][j]-d['phi'][i];y=(d['gains'][j]-d['gains'][i])/float(d['scale'])
 if mode=='interaction_only':X=X-X.mean(1,keepdims=True);y=y-y.mean(1,keepdims=True)
 return X.reshape(-1,85),y.ravel()
def summary_metric(rows):
 mse=float(np.mean([r['mse'] for r in rows]));energy=float(np.mean([r['target_energy'] for r in rows]));mean=float(np.mean([r['target_mean'] for r in rows]));return dict(events=len(rows),mse=mse,rmse=float(np.sqrt(mse)),zero_baseline_mse=energy,R2_vs_zero=1-mse/energy if energy>1e-30 else None,R2_centered=1-mse/(energy-mean**2) if energy-mean**2>1e-30 else None)
def fit():
 modeldir=OUT/'models';modeldir.mkdir(exist_ok=True)
 if (OUT/'model_lock.json').exists():return
 weights={};trials=[];selected=[]
 for mode in MODES:
  XX=np.zeros((85,85));XY=np.zeros(85)
  for eid in ids('train'):
   d=np.load(OUT/'events'/eid/'probes.npz');X,y=deltas(d,mode);XX+=X.T@X/len(y)/36;XY+=X.T@y/len(y)/36
  for family,sl in FAMILIES.items():
   indices=np.arange(85)[sl];A=XX[np.ix_(indices,indices)];b=XY[indices];rms=np.sqrt(np.maximum(np.diag(A),0));rms=np.maximum(rms,1e-12)
   for lam in LAMBDAS:
    coef=np.zeros(85);coef[indices]=np.linalg.solve(A/rms[:,None]/rms[None,:]+lam*np.eye(len(indices)),b/rms)/rms;key=f'{mode}__{family}__{lam}';weights[key]=coef
 # Development selection only, evaluate all predefined linear probes together.
 validation={key:[] for key in weights}
 for mode in MODES:
  keys=[k for k in weights if k.startswith(mode+'__')];C=np.stack([weights[k] for k in keys],1)
  for eid in ids('validation'):
   d=np.load(OUT/'events'/eid/'probes.npz');X,y=deltas(d,mode);pred=X@C
   for i,key in enumerate(keys):validation[key].append(dict(event_id=eid,mse=float(np.mean((pred[:,i]-y)**2)),target_energy=float(np.mean(y**2)),target_mean=float(y.mean())))
  for family in FAMILIES:
   kk=[k for k in keys if k.split('__')[1]==family];best=min(kk,key=lambda k:summary_metric(validation[k])['mse']);lam=float(best.split('__')[2]);coef=weights[best];name=f'{mode}__{family}';np.savez(modeldir/(name+'.npz'),coef=coef,lambda_=lam)
   selected.append(dict(name=name,mode=mode,family=family,lambda_=lam,development=summary_metric(validation[best]),weights_sha256=sha(modeldir/(name+'.npz'))))
   for key in kk:trials.append(dict(key=key,**summary_metric(validation[key])))
 # Static pair-only linear preference is shared by every correction.
 XX=np.zeros((21,21));XY=np.zeros(21)
 for eid in ids('train'):
  d=np.load(OUT/'events'/eid/'probes.npz');X=np.column_stack((d['pair'],np.ones(len(d['pair']))));y=(d['gains'][0]-d['gains'][0].mean())/float(d['scale']);XX+=X.T@X/len(y)/36;XY+=X.T@y/len(y)/36
 rms=np.maximum(np.sqrt(np.diag(XX)),1e-12);static=[]
 for lam in LAMBDAS:
  reg=np.eye(21)*lam;reg[-1,-1]=0;coef=np.linalg.solve(XX/rms[:,None]/rms[None,:]+reg,XY/rms)/rms;errors=[]
  for eid in ids('validation'):
   d=np.load(OUT/'events'/eid/'probes.npz');X=np.column_stack((d['pair'],np.ones(len(d['pair']))));y=(d['gains'][0]-d['gains'][0].mean())/float(d['scale']);errors.append(float(np.mean((X@coef-y)**2)))
  static.append((np.mean(errors),lam,coef))
 best=min(static,key=lambda x:x[0]);np.savez(modeldir/'static_pair.npz',coef=best[2],lambda_=best[1]);dump(OUT/'development_trials.json',trials)
 dump(OUT/'model_lock.json',dict(frozen_at=time.time(),train_events=ids('train'),validation_events=ids('validation'),models=selected,static_pair=dict(lambda_=best[1],development_mse=float(best[0]),sha256=sha(modeldir/'static_pair.npz')),evaluation_accessed=False,regularizers=LAMBDAS,target_scale='oracle event gain std used for labels/evaluation only; never inference feature',state_difference_models='no intercept; zero feature difference -> zero predicted change'))
 print(json.dumps(selected,indent=2))
if __name__=='__main__':fit()
