from .core import *
def fit():
 if (OUT/'model_lock.json').exists():return
 records=[]
 for rep in REPS:
  A=np.zeros((2,2));b=np.zeros(2);c=0
  for eid in ids('train'):
   d=np.load(OUT/'events'/eid/'terms.npz');ip=d[rep+'_I'];bp=d[rep+'_B'];scale=max(float(np.sqrt(np.mean(ip**2+bp**2))),1e-15);X=np.stack((ip,-bp),-1).reshape(-1,2)/scale;y=d['gains'].ravel()/max(float(d['gains'].std()),1e-12)
   A+=X.T@X/len(y)/36;b+=X.T@y/len(y)/36;c+=np.mean(y*y)/36
  candidates=[np.zeros(2),np.array([max(0,b[0]/A[0,0]),0]),np.array([0,max(0,b[1]/A[1,1])])];unconstrained=np.linalg.solve(A,b)
  if (unconstrained>=0).all():candidates.append(unconstrained)
  coef=min(candidates,key=lambda x:float(x@A@x-2*b@x));lam=float(coef[1]/coef[0]) if coef[0]>1e-15 else None
  trials=[]
  for name,value in [('baseline',1.),('fitted',lam)]:
   rows=[]
   for eid in ids('validation'):
    d=np.load(OUT/'events'/eid/'terms.npz');s=score(d,rep,value)[0];j=int(s.argmax());gain=float(d['gains'][0,j]) if s[j]>0 else 0.;rows.append(dict(event_id=eid,relative_gain=gain/float(d['nmse'][0]),gain=gain))
   trials.append(dict(name=name,lambda_=value,mean_relative_gain=float(np.mean([r['relative_gain'] for r in rows])),degradation_fraction=float(np.mean([r['gain']< -1e-12 for r in rows])),rows=rows))
  chosen=max(trials,key=lambda x:x['mean_relative_gain']);grad=A@coef-b;assert np.all(grad[coef<=1e-15]>=-1e-8) and np.linalg.norm(grad[coef>1e-15])<1e-8
  records.append(dict(representation=rep,lambda_fitted=lam,selected=chosen['name'],lambda_selected=chosen['lambda_'],coefficient=coef.tolist(),normal_matrix=A.tolist(),normal_rhs=b.tolist(),train_loss=float(coef@A@coef-2*b@coef+c),KKT_residual=grad.tolist(),development_trials=trials))
 dump(OUT/'model_lock.json',dict(frozen_at=time.time(),models=records,train_ids=ids('train'),development_ids=ids('validation'),evaluation_accessed=False));print(json.dumps([{k:v for k,v in r.items() if k!='development_trials'} for r in records],indent=2))
if __name__=='__main__':fit()
