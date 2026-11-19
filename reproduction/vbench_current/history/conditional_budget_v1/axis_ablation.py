import json,time
import numpy as np
from calibrate import d,train,tests
start=time.monotonic()
grid=json.loads((d/(train[0]+'.json')).read_text())
z=np.load(d/'calibration_curves.npz')
ss,en=z['sse'],z['energy']
cases=[]
for test in tests:
 with np.load(d/(test+'.npz')) as f:cases.append(f['sse'][2:,...,:54].astype(float))
rows=[]
for b,j in [(10,2),(15,4)]:
 for dims in ('','s','l','sl','lh','sh','slh'):
  sums=np.zeros(4)
  for i,k,w in ((0,1,216),(1,2,288),(2,3,360),(3,39,432)):
   e=ss[i:k].sum(0)/en[i:k].sum(0)[...,None]
   for axis,tag in enumerate('slh'):
    if tag not in dims:e=e.mean(axis=axis,keepdims=True)
   e=e.reshape(*e.shape[:-1],6,9)
   cost=np.array(grid['counts'][str(w)][:54:9]).sum(1)
   val=e.min(-1).mean(2,keepdims=True)
   lo,hi=0,max(1.,val.max())
   for _ in range(55):
    m=(lo+hi)/2; pick=(val+m*cost).argmin(-1)
    if cost[pick].mean()>cost[j]:lo=m
    else:hi=m
   pick=(val+hi*cost).argmin(-1)
   ix=9*pick+np.take_along_axis(e.argmin(-1),pick[...,None],axis=-1)[...,0]
   for n,x in enumerate(cases):sums[n]+=np.take_along_axis(x[i:k],ix[None,...,None],-1).sum()
  rows.append([b,dims,(sums/[x[...,j*9+2].sum() for x in cases]).tolist()])
(d/'axis_ablation.json').write_text(json.dumps(dict(rows=rows,tests=tests,seconds=time.monotonic()-start),indent=2))
for b,dims,r in rows:print(b,dims,round(100*(1-np.mean(r)),1))
