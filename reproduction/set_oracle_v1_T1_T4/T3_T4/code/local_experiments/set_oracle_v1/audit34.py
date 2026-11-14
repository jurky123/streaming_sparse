import json,tempfile,concurrent.futures
from pathlib import Path
import numpy as np
from .stage34 import ROOT,OUT,online_load,online_features,split
from .common import dump,sha

def check(eid):
 path=ROOT/'contributions'/eid;r=np.load(path/'residuals_fp64.npz');e=r['e'];C=float(np.square(r['dense_output']).sum()+1e-12);checks=0;maxerr=0.
 for stage in ('t3','t3_paired','t4'):
  data=json.loads((OUT/stage/f'{eid}.json').read_text())
  for method,result in data['results'].items():
   for tr in result['trace']:
    x=np.zeros(e.shape[1],bool);x[tr['selected']]=True
    assert x.sum()==int(r['K']) and x[int(r['own'])] and len(tr['selected'])==int(r['K'])
    E=e[:,~x].sum(1);Z=np.where(x[None],r['exact_mass'],r['summary_mass']).sum(1);j=float(np.sum(np.square(E).sum(-1)/Z**2)/C)
    err=abs(j-tr['nmse']);assert err<max(1e-11,abs(j)*1e-8);maxerr=max(maxerr,err);checks+=1
 f=online_load(path/'summary_and_route_features.npz');W=np.load(OUT/'projections.npz')['feature'];X,mass=online_features(f,W);truth=r['summary_mass']/r['summary_mass'].sum(1,keepdims=True);masserr=float(np.max(np.abs(mass-truth)));assert masserr<1e-10;assert np.isfinite(X).all()
 return dict(event_id=eid,masks_checked=checks,max_objective_error=maxerr,max_online_summary_mass_error=masserr)
def run():
 ids=sorted(p.name for p in (ROOT/'contributions').iterdir())
 with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:rows=list(ex.map(check,ids))
 expected=json.loads((ROOT/'delivery_checksums.json').read_text());changed=[]
 def hashed(k):return k,sha(ROOT/k)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for k,h in ex.map(hashed,expected):
   if h!=expected[k]:changed.append(k)
 assert not changed
 # Explicit poisoning test: allowlist features must not depend on teacher fields.
 path=ROOT/'contributions'/ids[0]/'summary_and_route_features.npz';data=dict(np.load(path));before=online_load(path)
 for key in ('exact_page_logz','summary_page_logz','online_summary_logz'):
  if key in data:data[key]=np.full_like(data[key],np.nan)
 with tempfile.TemporaryDirectory() as td:
  f=Path(td)/'poison.npz';np.savez(f,**data);after=online_load(f)
 for key in before:assert np.array_equal(before[key],after[key])
 proj=np.load(OUT/'projections.npz');assert np.max(np.abs(proj['pca'].T@proj['pca']-np.eye(128)))<1e-10
 train_ids=[e for e in ids if split(e)=='train'];val=[e for e in ids if split(e)=='validation'];test=[e for e in ids if split(e)=='evaluation'];assert len(train_ids)==len(val)==36 and len(test)==72
 result=dict(passed=True,instances=144,total_masks_checked=sum(r['masks_checked'] for r in rows),max_objective_error=max(r['max_objective_error'] for r in rows),max_summary_mass_error=max(r['max_online_summary_mass_error'] for r in rows),frozen_files_verified=len(expected),changed_frozen_files=changed,teacher_poisoning_test='passed; exact_page_logz and reference logZ not accessed by online allowlist',splits=dict(train=train_ids,validation=val,evaluation=test),rows=rows)
 dump(OUT/'audit.json',result);print({k:v for k,v in result.items() if k not in ('rows','splits')})
if __name__=='__main__':run()
