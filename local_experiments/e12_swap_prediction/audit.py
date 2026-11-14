import json,tempfile,concurrent.futures,time
import numpy as np
import torch
from .core import *
from .evaluate import model_scores

def check(eid,rows):
 path=DATA/'contributions'/eid;f=load_online(path);p=pages(f);saved=np.load(OUT/'states'/f'{eid}.npz');x=initial_mask(p,int(saved['own']),int(saved['K']));F,r,a=features(p,x,int(saved['own']));assert np.array_equal(F,saved['features']) and np.array_equal(r,saved['remove']) and np.array_equal(a,saved['add'])
 with np.load(path/'residuals_fp64.npz') as raw:
  raw=dict(raw);direct=swap_labels(raw,x,r,a);err=float(np.max(np.abs(direct-saved['gains'])));assert err<1e-10
  for row in rows:
   mask=np.zeros(len(x),bool);mask[row['selected']]=True;assert mask[int(saved['own'])] and mask.sum()==int(saved['K']);true=objective(raw,mask);assert abs(true-row['nmse'])<1e-10
 return dict(event_id=eid,max_label_error=err,masks_checked=len(rows),scalar_changes=bool(saved['scalar_changed']),direction_changes=bool(saved['directional_changed']))
def run():
 torch.set_num_threads(1);rows=json.loads((OUT/'evaluation_per_instance.json').read_text());ids=sorted(p.stem for p in (OUT/'states').glob('*.npz'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:checks=list(ex.map(lambda e:check(e,[r for r in rows if r['event_id']==e]),ids))
 # Ensure only training instances determine standardization; same capacity/initial weights.
 train=np.stack([np.load(OUT/'states'/f'{e}.npz')['features'] for e in ids if split(e)=='train']);means=train.mean((0,1));std=np.maximum(train.std((0,1)),1e-6);initial=[]
 for arch in ('A0','A1','A2'):
  name=arch+'_s17';w=np.load(OUT/'models'/name/'weights.npz');assert np.array_equal(means,w['mean']) and np.array_equal(std,w['std']);meta=json.loads((OUT/'models'/name/'training.json').read_text());assert meta['updates']==2000 and meta['parameters']==1089;initial.append(meta['initial_weight_sha256']);assert all(split(e)=='train' for e in meta['train_states']) and all(split(e)=='validation' for e in meta['validation_states'])
 assert len(set(initial))==1
 # Teacher poisoning: arbitrary NaNs in all non-allowlist arrays cannot affect page features.
 path=DATA/'contributions'/ids[0];original=dict(np.load(path/'summary_and_route_features.npz'));base=pages(load_online(path))
 for key in original:
  if key not in ALLOW:
   if np.issubdtype(original[key].dtype,np.floating):original[key]=np.full_like(original[key],np.nan)
   else:original[key]=np.zeros_like(original[key])
 with tempfile.TemporaryDirectory() as td:
  np.savez(Path(td)/'summary_and_route_features.npz',**original);poisoned=pages(load_online(Path(td)))
 for k in base:assert np.array_equal(base[k],poisoned[k])
 # Archived labels/file hashes remain unchanged from T1/T2 freeze.
 expected=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for name,h in ex.map(lambda name:(name,sha(DATA/name)),expected):assert h==expected[name],name
 lock=json.loads((OUT/'evaluation_lock.json').read_text())
 for model in lock['models']:assert sha(OUT/'models'/model['name']/'weights.npz')==model['weights_sha256']
 report=dict(passed=True,instances=144,independent_action_labels=144*512,max_label_error=max(r['max_label_error'] for r in checks),evaluation_masks_checked=sum(r['masks_checked'] for r in checks),scalar_changes=sum(r['scalar_changes'] for r in checks),direction_changes=sum(r['direction_changes'] for r in checks),same_initial_weights=True,same_updates=2000,same_parameters=1089,train_only_normalizer=True,teacher_poisoning_pass=True,frozen_files_verified=len(expected),checks=checks)
 dump(OUT/'audit.json',report);print({k:v for k,v in report.items() if k!='checks'})
if __name__=='__main__':run()
