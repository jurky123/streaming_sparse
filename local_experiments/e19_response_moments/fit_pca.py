from .core import *
def one(eid):
 _,_,cs=components(eid);out={}
 for n,v in cs.items():
  X=v.reshape(-1,128);m=X.T@X;out[n]=m/max(float(np.trace(m)),1e-30)
 return eid,out
if __name__=='__main__':
 preregister()
 if not (OUT/'pca_lock.json').exists():
  sums={n:np.zeros((128,128)) for n in ['net','V','Z']}
  with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
   for eid,ms in ex.map(one,ids('train')):
    for n in sums:sums[n]+=ms[n]/36
  models={};meta=[]
  for n,m in sums.items():
   val,U=np.linalg.eigh(m);order=np.argsort(val)[::-1];U=U[:,order[:16]];assert np.allclose(U.T@U,np.eye(16),atol=1e-12);models[n]=U;models[n+'_spectrum']=val[order];meta.append(dict(component=n,training_retained_normalized_energy=float(val[order[:16]].sum()/val.sum())))
  np.savez(OUT/'pca16.npz',**models)
  dump(OUT/'pca_lock.json',dict(frozen_at=time.time(),train_ids=ids('train'),models=meta,sha256=sha(OUT/'pca16.npz'),evaluation_accessed=False));print(meta)
