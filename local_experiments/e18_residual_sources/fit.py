from .core import *
def run():
 if (OUT/'calibration_lock.json').exists():return
 old=json.loads((E17/'model_lock.json').read_text());models=[]
 for rep in REPS:
  lam=next(m['lambda_fitted'] for m in old['models'] if m['representation']==rep);assert lam is not None
  cross=den=target=0.;records=[]
  for eid in ids('train'):
   d=np.load(OUT/'events'/eid/'terms.npz');x=d[rep+'_unit_I'];y=lam*d[rep+'_unit_B'];I=d['I'];B=d['B'];scale=max(float(np.mean(I**2+B**2)),1e-24)
   num=float(np.mean(x*I+y*B)/scale);norm=float(np.mean(x*x+y*y)/scale);cross+=num/36;den+=norm/36;target+=float(np.mean(I*I+B*B)/scale)/36;records.append(dict(event_id=eid,cross=num,norm=norm,target_scale=scale))
  c=max(0.,cross/den);assert c>0
  models.append(dict(representation=rep,E17_lambda=lam,common_scale=c,cross=cross,norm=den,train_objective=float(c*c*den-2*c*cross+target),train_records=records))
 dump(OUT/'calibration_lock.json',dict(created=time.time(),models=models,train_ids=ids('train'),no_validation_selection=True,no_evaluation_access=True,E17_model_sha=sha(E17/'model_lock.json')))
 print([{k:v for k,v in m.items() if k!='train_records'} for m in models])
if __name__=='__main__':run()
