from calibrate import d,train
import json,numpy as np
grid=json.loads((d/(train[0]+'.json')).read_text())
tabs=np.load(d/'policies.npz'); rows=[]
for budget in (10,15):
 for label in ('window_step_layer_head','independent_heads'):
  work=padded=mean_pages=0.
  tab=tabs[f'{budget}_{label}']
  for i,c in enumerate(range(2,41)):
   w=min(432,72*(c+1));n=24+np.array(grid['counts'][str(w)]).sum(1)[tab[i]]
   constant=2*4608*((w+15)//16*16)*(4*128+2*128+2*3*16)
   exact=6*4608*64*128
   work+=(n*exact+constant).sum()
   padded+=(n.max(-1)*exact+constant).sum()*12
   mean_pages+=n.mean()
  rows.append(dict(budget=budget,schedule=label,mean_exact=mean_pages/39,ideal_tflop=work/1e12,padded_tflop=padded/1e12,padding_factor=padded/work))
(d/'ragged_cost.json').write_text(json.dumps(rows,indent=2))
