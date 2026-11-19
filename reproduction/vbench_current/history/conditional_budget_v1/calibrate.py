import json,time,csv
from pathlib import Path
import numpy as np
d=Path(__file__).parent
train=['forest_run_113','forest_run_557','kitchen_cook_113','kitchen_cook_557']
tests=['snow_ski_113','forest_run_917','snow_ski_917','bird_flight_919']
def main():
 start=time.monotonic()
 grid=json.loads((d/(train[0]+'.json')).read_text())
 def read(name):
    with np.load(d/(name+'.npz')) as f:return f['sse'][2:,...,:54].astype(float),f['dense_energy'][2:].astype(float)
 ss,en=map(sum,zip(*(read(x) for x in train)))
 results=[]; tables={}
 for budget in (10,15):
  anchor=(2 if budget==10 else 4)*9+2
  for gran in ('window','window_step_layer','window_step_layer_head','full_age','chunk','independent_heads'):
   table=np.zeros(ss.shape[:-1],int)
   for earliest,last in [(2,3),(3,4),(4,5)]+([(5,41)] if gran not in ('full_age','chunk') else ([(5,6),(6,12),(12,24),(24,41)] if gran=='full_age' else [(i,i+1) for i in range(5,41)])):
    sl=slice(earliest-2,last-2);w=min(432,(earliest+1)*72);quota=np.array(grid['counts'][str(w)][:54]);cost=quota.sum(1)
    err=ss[sl].sum(0)/en[sl].sum(0)[...,None]
    if gran=='window':err=err.mean((0,1,2),keepdims=True)
    if gran=='window_step_layer':err=err.mean(2,keepdims=True)
    # Each fused call needs a common exact length; allocation within that length may differ by head.
    best=err.reshape(*err.shape[:-1],6,9).argmin(-1)
    vals=err.reshape(*err.shape[:-1],6,9).min(-1)
    if gran!='independent_heads':vals=vals.mean(2,keepdims=True)
    levels=cost[::9]
    lo,hi=0.,max(1.,vals.max())
    for _ in range(55):
     mid=(lo+hi)/2;sel=(vals+mid*levels).argmin(-1)
     if levels[sel].mean()>cost[anchor]:lo=mid
     else:hi=mid
    sel=(vals+hi*levels).argmin(-1)
    sel=np.broadcast_to(sel,best.shape[:-1])
    table[sl]=sel*9+np.take_along_axis(best,sel[...,None],axis=-1)[...,0]
   tables[f'{budget}_{gran}']=table.astype('uint8')
   for name in tests:
    a,b=read(name);x=np.take_along_axis(a,table[...,None],axis=-1)[...,0]
    results.append(dict(budget=budget,gran=gran,test=name,ratio=x.sum()/a[...,anchor].sum(),nmse=x.sum()/b.sum(),per_chunk=(x.sum((1,2,3))/b.sum((1,2,3))).tolist()))
 np.savez_compressed(d/'policies.npz',**tables)
 (d/'holdouts.json').write_text(json.dumps(results,indent=2))
 with (d/'quotas.csv').open('w') as f:
  wr=csv.writer(f);wr.writerow(['W','grid_id','total_nominal','exact_current','exact_history','p','q','realized_total'])
  for w,entries in grid['counts'].items():
   for i,(a,b) in enumerate(entries):wr.writerow([w,i,grid['grid'][i][0],a,b,a/72,b/(int(w)-96),(a+b)/(int(w)-24)])
 (d/'calibration_seconds').write_text(str(time.monotonic()-start))
if __name__=='__main__':main()
