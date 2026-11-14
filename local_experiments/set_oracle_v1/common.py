import json,hashlib,os
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/set_oracle_v1')
def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def head(layer,chunk,seed,step):return (layer+chunk+(seed-101)//202+3*step)%12
def event(case,seed,c,l,s,ti):
 h=head(l,c,seed,s);t=((17*l+13*c+seed)%72+36*ti)%72
 return dict(event_id=f'{case}_s{seed}_c{c}_l{l}_s{s}_h{h}_t{t}',case=case,seed=seed,chunk=c,layer=l,step=s,head=h,tile=t,tile_index=ti,P=216 if c==2 else 410,K=21 if c==2 else 41,Q=64)
def events():return [event(case,seed,c,l,s,ti) for case in ('city_walk','horse_field') for seed in (101,303) for c in (2,5,8) for l in (1,15,29) for s in (0,2) for ti in (0,1)]
def direct(d,x,queries=None):
 q=np.arange(64) if queries is None else queries;N,Z,Nh,Zh=(d[k][q] for k in ('N','Z','Nh','Zh'));od=N.sum(1)/Z.sum(1)[:,None];out=np.where(x[None,:,None],N,Nh).sum(1)/np.where(x[None],Z,Zh).sum(1)[:,None]
 return float(np.square(out-od).sum()/(np.square(od).sum()+1e-12))
