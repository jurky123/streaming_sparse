import concurrent.futures,tempfile
import numpy as np
from .core import *
from .train import load,event_normalizer

def one(eid,rows):
 root=DATA/'contributions'/eid;raw=dict(np.load(root/'residuals_fp64.npz'));f=load_online(root);p=pages(f);s=np.load(OUT/'events'/eid/'states.npz');an=np.load(OUT/'events'/eid/'anchors.npz');errors=[];mask_count=0
 for i,x in enumerate(s['masks']):
  assert x[int(s['own'])] and x.sum()==int(s['K']);r,a=pool(p,x,int(s['own']));F=pair_features(p,x,r,a);assert np.array_equal(F,s['features'][i]);assert np.array_equal(r,s['remove'][i]) and np.array_equal(a,s['add'][i]);g=swap_labels(raw,x,r,a);errors.append(float(np.max(np.abs(g-s['gains'][i]))));assert errors[-1]<1e-10
  # Independent full-mask objective for 8 fixed candidate IDs in each state.
  for idx in [0,1,17,63,127,255,383,511]:
   y=x.copy();y[r[idx]]=False;y[a[idx]]=True;err=abs(objective(raw,x)-objective(raw,y)-g[idx]);assert err<1e-10;errors.append(err)
  ar,aa=an['remove'],an['add'];assert np.all(x[ar]) and np.all(~x[aa]);ag=swap_labels(raw,x,ar,aa);err=float(np.max(np.abs(ag-an['gains'][i])));assert err<1e-10;errors.append(err)
 for row in rows:
  x=np.zeros(s['masks'].shape[1],bool);x[row['selected']]=True;assert x[int(s['own'])] and x.sum()==int(s['K']);j=objective(raw,x);assert abs(j-row['nmse_after'])<max(1e-12,j*1e-9);mask_count+=1
 return dict(event_id=eid,states=len(s['masks']),medium_labels=int(s['gains'].size),anchor_labels=int(an['gains'].size),max_error=max(errors),evaluation_masks=mask_count)
def run():
 ids=sorted(p.name for p in (OUT/'events').iterdir());rows=json.loads((OUT/'evaluation_per_state.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:checks=list(ex.map(lambda e:one(e,[r for r in rows if r['event_id']==e]),ids))
 initial=[];sampler={}
 for name in ('A0_Risk','A2_Risk','A0_CF','A2_CF'):
  tr=json.loads((OUT/'models'/name/'training.json').read_text());initial.append(tr['initial_weight_sha256']);assert tr['parameters']==1089 and tr['updates']==2000;assert all(split(e)=='train' for e in tr['train_events']);weights=np.load(OUT/'models'/name/'weights.npz');_,X,_,_,_,counts=load('train',tr['regime']);mean,std=event_normalizer(X,counts);assert np.array_equal(mean,weights['mean']) and np.array_equal(std,weights['std']);sampler[name]=tr['sampler_sha256']
 assert len(set(initial))==1;assert sampler['A0_CF']==sampler['A2_CF'] and sampler['A0_Risk']==sampler['A2_Risk']
 # Higher precision audit of worst relative degradation, without removing it.
 worst=min((r for r in rows if r['state']==0),key=lambda r:r['gain_pct']);raw=dict(np.load(DATA/'contributions'/worst['event_id']/'residuals_fp64.npz'));s=np.load(OUT/'events'/worst['event_id']/'states.npz');vals=[]
 for mask in [s['masks'][0],np.isin(np.arange(s['masks'].shape[1]),worst['selected'])]:
  E=raw['e'].astype(np.longdouble)[:,~mask].sum(1);Z=np.where(mask[None],raw['exact_mass'].astype(np.longdouble),raw['summary_mass'].astype(np.longdouble)).sum(1);C=np.square(raw['dense_output'].astype(np.longdouble)).sum()+np.longdouble(1e-12);vals.append(float((np.square(E).sum(1)/np.square(Z)).sum()/C))
 assert np.allclose(vals,[worst['risk_or_state_nmse'],worst['nmse_after']],rtol=1e-10,atol=1e-16)
 frozen=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for name,h in ex.map(lambda name:(name,sha(DATA/name)),frozen):assert h==frozen[name]
 # Input allowlist poisoning, no forbidden teacher arrays reach features.
 path=DATA/'contributions'/ids[0];f0=load_online(path);allf=dict(np.load(path/'summary_and_route_features.npz'))
 for k in allf:
  if k not in ALLOW:allf[k]=np.zeros_like(allf[k])
 with tempfile.TemporaryDirectory() as td:
  np.savez(Path(td)/'summary_and_route_features.npz',**allf);f1=load_online(Path(td))
 for k in f0:assert np.array_equal(f0[k],f1[k])
 result=dict(passed=True,events=len(ids),states=sum(c['states'] for c in checks),medium_labels=sum(c['medium_labels'] for c in checks),anchor_labels=sum(c['anchor_labels'] for c in checks),max_error=max(c['max_error'] for c in checks),evaluation_masks=sum(c['evaluation_masks'] for c in checks),same_initial_weights=True,same_updates=2000,same_parameters=1089,equal_event_batch=True,train_only_normalizer=True,teacher_allowlist_pass=True,frozen_files_verified=len(frozen),worst_case_high_precision=dict(event_id=worst['event_id'],model=worst['model'],baseline_nmse=vals[0],after_nmse=vals[1],relative_gain_pct=100*(vals[0]-vals[1])/vals[0],precision_bits=int(np.finfo(np.longdouble).nmant)),checks=checks)
 dump(OUT/'audit.json',result);print({k:v for k,v in result.items() if k!='checks'})
if __name__=='__main__':run()
