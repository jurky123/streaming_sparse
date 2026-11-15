from .core import *
from ..e12_swap_prediction.core import ALLOW,swap_labels
import tempfile

def one(eid):
 from .prepare import energy
 r=json.loads((OUT/'events'/eid/'result.json').read_text());d=np.load(OUT/'events'/eid/'terms.npz');assert sha(OUT/'events'/eid/'terms.npz')==r['audit']['terms_sha']
 for path,h in r['audit']['sources'].items():assert sha(ROOT/path)==h
 st=np.load(E13/'events'/eid/'states.npz');raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));G=d['gains'];lock=json.loads((OUT/'calibration_lock.json').read_text());scores=[G,d['I']-d['B']]
 old=np.load(E17/'events'/eid/'terms.npz')
 for model in lock['models']:
  rep=model['representation'];c=model['common_scale'];lam=model['E17_lambda'];ip=c*d[rep+'_unit_I'];bp=c*lam*d[rep+'_unit_B'];sc=ip-bp;baseline=old[rep+'_I']-lam*old[rep+'_B'];assert np.array_equal(sc.argmax(1),baseline.argmax(1));assert np.array_equal(sc.max(1)>0,baseline.max(1)>0)
  scores.extend([sc,d['I']-bp,ip-d['B'],4*d['I16']-bp,ip-4*d['B16']])
 err=0
 for i,x in enumerate(st['masks']):
  js=np.unique([int(s[i].argmax()) for s in scores]+[0,G.shape[1]-1]);vv=swap_labels(raw,x,d['remove'][js],d['add'][js]);err=max(err,float(np.max(abs(vv-G[i,js]))))
 assert err<1e-10
 return dict(event_id=eid,max_label_error=err,**{k:v for k,v in r['audit'].items() if k not in ['sources','terms_sha']})
def run():
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:rows=list(ex.map(one,ids()))
 lock=json.loads((OUT/'calibration_lock.json').read_text());assert lock['train_ids']==ids('train');assert lock['E17_model_sha']==sha(E17/'model_lock.json')
 calibration=[]
 for model in lock['models']:
  cross=norm=0
  for eid in ids('train'):
   d=np.load(OUT/'events'/eid/'terms.npz');x=d[model['representation']+'_unit_I'];y=model['E17_lambda']*d[model['representation']+'_unit_B'];I=d['I'];B=d['B'];w=max(float(np.mean(I**2+B**2)),1e-24);cross+=float(np.mean(x*I+y*B)/w)/36;norm+=float(np.mean(x*x+y*y)/w)/36
  assert abs(max(0,cross/norm)-model['common_scale'])<1e-12
  for sp in ['train','validation','evaluation']:
   vals=[]
   for eid in ids(sp):
    d=np.load(OUT/'events'/eid/'terms.npz');ip=model['common_scale']*d[model['representation']+'_unit_I'];bp=model['common_scale']*model['E17_lambda']*d[model['representation']+'_unit_B'];I=d['I'];B=d['B'];w=max(float(np.mean(I**2+B**2)),1e-24);vals.append(dict(error=float(np.mean((ip-I)**2+(bp-B)**2)/w),I_mse=float(np.mean((ip-I)**2)),B_mse=float(np.mean((bp-B)**2))))
   calibration.append(dict(split=sp,representation=model['representation'],normalized_joint_mse=float(np.mean([v['error'] for v in vals])),zero_predictor_normalized_mse=1.0))
 f=load_online(DATA/'contributions'/ids()[0]);full=dict(np.load(DATA/'contributions'/ids()[0]/'summary_and_route_features.npz'))
 for k in full:
  if k not in ALLOW:full[k]=np.zeros_like(full[k])
 with tempfile.TemporaryDirectory() as td:
  np.savez(Path(td)/'summary_and_route_features.npz',**full);ff=load_online(Path(td))
 a,b=summary_output(f);aa,bb=summary_output(ff);assert np.array_equal(a,aa) and np.array_equal(b,bb)
 original=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for name,h in ex.map(lambda n:(n,sha(DATA/n)),original):assert h==original[name]
 n=0
 for event in json.loads((E13/'preparation.json').read_text())['events']:
  for name,h in event['files'].items():assert sha(E13/'events'/event['event_id']/name)==h;n+=1
 assert json.loads((OUT/'summary.json').read_text())['calibration_lock_sha']==sha(OUT/'calibration_lock.json')
 result=dict(passed=True,events=len(rows),source_files=len(original),E13_files=n,source_identity_pass=True,baseline_E17_masks_and_gate_unchanged=True,teacher_poisoning_pass=True,training_only_common_calibration_pass=True,max_label_error=max(r['max_label_error'] for r in rows),max_summary_global_identity_error=max(r['summary_global_identity_error'] for r in rows),max_interaction_decomposition_error=max(r['I_decomposition_error'] for r in rows),max_set_energy_error=max(r['set_energy_error'] for r in rows),calibration_diagnostics=calibration)
 dump(OUT/'audit.json',result);print(result)
if __name__=='__main__':run()
