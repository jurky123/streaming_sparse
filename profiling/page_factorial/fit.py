import sys,json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent/'calibration'/sys.argv[1]
train=['forest_run_113','forest_run_557','kitchen_cook_113','kitchen_cook_557']
test=['snow_ski_917','bird_flight_919']
def read(n):
    with np.load(p/(n+'.npz')) as f:return f['sse'][2:,...,:54].astype(float),f['dense_energy'][2:].astype(float)
s,e=map(sum,zip(*(read(n) for n in train)))
grid=json.loads((p/(train[0]+'.json')).read_text())['counts']
table=np.zeros(s.shape[:-1],dtype=np.uint8)
log=[]
for a,b in ((0,1),(1,2),(2,3),(3,5)):
    w=min(432,(a+3)*72)
    cost=np.sum(grid[str(w)][:54],axis=1)
    err=s[a:b].sum(0)/e[a:b].sum(0)[...,None]
    for steps in (slice(0,4),slice(4,5)):
        x=err[steps].reshape(-1,29,12,6,9)
        share=x.argmin(-1);v=x.min(-1).mean(2,keepdims=True)
        lo,hi=0.,max(1.,v.max())
        for _ in range(55):
            mid=(hi+lo)/2
            level=(v+mid*cost[::9]).argmin(-1)
            if cost[::9][level].mean()>cost[20]:lo=mid
            else:hi=mid
        level=np.broadcast_to((v+hi*cost[::9]).argmin(-1),share.shape[:-1])
        ids=level*9+np.take_along_axis(share,level[...,None],axis=-1)[...,0]
        table[a:b,steps]=ids
        log.append([w,steps.start,float(cost[ids].mean()),int(cost[20])])
np.save(p/'table.npy',table)
rows=[]
for n in test:
    s,e=read(n)
    rows.append([n,float(np.take_along_axis(s,table[...,None],axis=-1).sum()/e.sum())])
(p/'fit.json').write_text(json.dumps(dict(costs=log,holdout_nmse=rows)))
