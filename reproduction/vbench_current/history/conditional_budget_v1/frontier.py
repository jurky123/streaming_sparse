from calibrate import d,train,tests
import numpy as np,csv,json
curve=np.load(d/'calibration_curves.npz')
grid=json.loads((d/(train[0]+'.json')).read_text())
s=curve['sse'];den=curve['energy']
with (d/'frontier.csv').open('w') as f:
 out=csv.writer(f);out.writerow(['window','nominal_average_cap','mean_exact','test','calibrated_nmse','fixed_nmse','ratio'])
 for lo,hi,w in ((0,1,216),(1,2,288),(2,3,360),(3,39,432)):
  vals=(s[lo:hi].sum(0)/den[lo:hi].sum(0)[...,None]).reshape(4,29,12,6,9)
  split=vals.argmin(-1)
  err=vals.min(-1).mean(2)
  levels=np.array(grid['counts'][str(w)][:54:9]).sum(1)
  for j,total in enumerate([5,8,10,12,15,20]):
   left,right=0,max(1.,err.max())
   for _ in range(55):
    mid=(left+right)/2; sel=(err+mid*levels).argmin(-1)
    if levels[sel].mean()>levels[j]:left=mid
    else:right=mid
   sel=(err+right*levels).argmin(-1);ix=sel[:,:,None]*9+np.take_along_axis(split,sel[:,:,None,None],axis=-1)[...,0]
   for name in tests:
    with np.load(d/(name+'.npz')) as z:
     a=z['sse'][2+lo:2+hi].astype(float).sum(0);b=z['dense_energy'][2+lo:2+hi].sum(dtype=float)
    n=np.take_along_axis(a,ix[...,None],-1).sum()/b;fixed=a[...,9*j+2].sum()/b
    out.writerow([w,total,24+levels[sel].mean(),name,n,fixed,n/fixed])
