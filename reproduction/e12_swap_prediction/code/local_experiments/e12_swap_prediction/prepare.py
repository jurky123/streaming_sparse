import concurrent.futures,json,time
import numpy as np
from .core import *
def run(eid):
 path=DATA/'contributions'/eid;out=OUT/'states'/f'{eid}.npz'
 if out.exists():return eid
 f=load_online(path);p=pages(f)
 with np.load(path/'residuals_fp64.npz') as z:own=int(z['own']);K=int(z['K']);risk=z['risk'];raw={k:z[k] for k in ('e','d','exact_mass','summary_mass','dense_output')}
 x=initial_mask(p,own,K);assert np.array_equal(x,risk);F,r,a=features(p,x,own)
 # Reuse saved complete initial exchange surface; no new raw-KV/attention computation.
 with np.load(DATA/'searches/v1'/eid/'candidate_gain_surfaces.npz') as surfaces:
  assert np.array_equal(surfaces['state_masks'][0],x);rr=surfaces['removed'][0];aa=surfaces['added'][0];g=surfaces['gains'][0][np.searchsorted(rr,r),np.searchsorted(aa,a)]
 direct=swap_labels(raw,x,r,a);err=float(np.max(np.abs(direct-g)));assert err<1e-10
 # Counterfactual state: same (remove,add), change two OTHER memberships.
 r0,a0=r[0],a[0];sel=np.flatnonzero(x);other_remove=next(int(k) for k in sel if k not in (own,r0));other_add=next(int(k) for k in np.flatnonzero(~x) if k!=a0);y=x.copy();y[other_remove]=False;y[other_add]=True
 s,H=state(p,x);inc=(s-p['additive'][other_remove]+p['additive'][other_add],H-p['H'][other_remove]+p['H'][other_add]);exact=state(p,y);assert np.max(np.abs(inc[0]-exact[0]))<1e-10 and np.max(np.abs(inc[1]-exact[1]))<1e-10
 # Compute same pair features by copying its online page descriptors and each set state.
 def single(st):
  ss,HH=st;n=ss[5];scalar=np.array([ss[0],ss[1],max(0,ss[3]/n-(ss[2]/n)**2),ss[4]/n]);hr=p['H'][r0];ha=p['H'][a0];hn=np.linalg.norm(HH);rn=np.linalg.norm(hr);an=np.linalg.norm(ha);dr=hr@HH;da=ha@HH;ch=ha-hr
  return np.r_[p['page'][r0],p['page'][a0],scalar,hn,dr,da,dr/(hn*rn+1e-30),da/(hn*an+1e-30),hr@ha/(rn*an+1e-30),np.linalg.norm(HH+ch)-hn,ch@HH/(np.linalg.norm(ch)*hn+1e-30)]
 fx,fy=single((s,H)),single(exact);assert np.allclose(fx,F[0],atol=1e-12);assert np.array_equal(fx[:20],fy[:20]);scalar_changed=bool(not np.allclose(fx[20:24],fy[20:24],rtol=1e-10,atol=1e-12));direction_changed=bool(not np.allclose(fx[24:],fy[24:],rtol=1e-10,atol=1e-12))
 out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,features=F,gains=g,remove=r,add=a,mask=x,own=own,K=K,baseline_nmse=objective(raw,x),label_direct_error=err,scalar_changed=scalar_changed,directional_changed=direction_changed)
 return eid
if __name__=='__main__':
 preregister();ids=sorted(p.name for p in (DATA/'contributions').iterdir())
 with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:
  for i,eid in enumerate(ex.map(run,ids),1):print(i,eid,flush=True)
 dump(OUT/'preparation.json',dict(complete=144,states_train=36,actions_per_state=512,no_op=True,source='frozen T2 initial candidate_gain_surfaces; independently checked against FP64 residual objective',states={e:sha(OUT/'states'/f'{e}.npz') for e in ids}))
