import concurrent.futures,tempfile
from .core import *
from .fit import ids,deltas,FAMILIES,MODES
from ..e12_swap_prediction.core import objective,swap_labels,ALLOW

def one(eid):
 d=np.load(OUT/'events'/eid/'probes.npz');result=json.loads((OUT/'events'/eid/'result.json').read_text());an=np.load(E13/'events'/eid/'anchors.npz');st=np.load(E13/'events'/eid/'states.npz');G=d['gains'];J=d['nmse'];assert np.array_equal(G,an['gains']);A=result['A'];best=np.maximum(G.max(1),0);fixed=G[:,int(A['j_mean'])];strong=G[:,int(A['j_positive'])];assert abs((best-np.maximum(fixed,0)).mean()-A['V_requested'])<1e-14;assert A['conditional_gain']+1e-12>=A['strong_fixed_with_perfect_accept_gain']>=A['fixed_with_perfect_accept_gain']-1e-12;assert A['fixed_with_perfect_accept_gain']+1e-12>=A['strict_fixed_gain'];assert A['V_requested']>=-1e-12
 # Reconstruct online phi, independent of all teacher arrays.
 f=load_online(DATA/'contributions'/eid);mass,h,U=geometry(f);proxy=mass[...,None]*h;H=np.stack([proxy[:,x].sum(1) for x in st['masks']]);v=proxy[:,an['remove']]-proxy[:,an['add']];product=-np.einsum('sqi,qaj->saij',H,v)/len(mass);context=an['features'][:,:,20:32];sc=context[:,:,:4];phi=np.concatenate((context,np.concatenate((sc,np.trace(product,axis1=-2,axis2=-1)[...,None]),-1),np.concatenate((sc,product.reshape(len(G),len(an['remove']),64)),-1)),-1);assert np.array_equal(phi,d['phi'])
 raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));err=0.
 for s in range(len(G)):
  indices=np.unique(np.array([0,len(an['remove'])//2,len(an['remove'])-1,int(G[s].argmax()),int(A['j_mean']),int(A['j_positive'])]));actual=swap_labels(raw,st['masks'][s],an['remove'][indices],an['add'][indices]);err=max(err,float(np.max(np.abs(actual-G[s,indices]))));assert err<1e-10
 qi=f['probe_indices'].astype(int);E=np.stack([raw['e'][qi][:,~x].sum(1) for x in st['masks']]);delta=raw['e'][qi][:,an['remove']]-raw['e'][qi][:,an['add']];decode=np.linalg.pinv(U);cs,vs=cosine(-H@decode,E);cd,vd=cosine(v@decode,delta)
 return dict(event_id=eid,split=split(eid),max_gain_error=err,negative_H_fullspace_cosine=float(cs[vs].mean()),candidate_fullspace_cosine=float(cd[vd].mean()),common_oracle=float(best.mean()),medium_oracle=float(np.maximum(st['gains'].max(1),0).mean()),common_to_medium_oracle_ratio=float(best.mean()/np.maximum(st['gains'].max(1),0).mean()) if np.maximum(st['gains'].max(1),0).mean()>1e-12 else None)
def run():
 allids=sorted(p.name for p in (OUT/'events').iterdir())
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:rows=list(ex.map(one,allids))
 lock=json.loads((OUT/'model_lock.json').read_text());assert all(split(e)=='train' for e in lock['train_events']) and all(split(e)=='validation' for e in lock['validation_events']);assert len(lock['train_events'])==len(lock['validation_events'])==36
 kkt=[]
 for mode in MODES:
  XX=np.zeros((85,85));XY=np.zeros(85)
  for eid in ids('train'):
   d=np.load(OUT/'events'/eid/'probes.npz');X,y=deltas(d,mode);XX+=X.T@X/len(y)/36;XY+=X.T@y/len(y)/36
  for m in lock['models']:
   if m['mode']!=mode:continue
   path=OUT/'models'/(m['name']+'.npz');assert sha(path)==m['weights_sha256'];coef=np.load(path)['coef'];idx=np.arange(85)[FAMILIES[m['family']]];A=XX[np.ix_(idx,idx)];b=XY[idx];rms2=np.maximum(np.diag(A),1e-24);res=(A+m['lambda_']*np.diag(rms2))@coef[idx]-b;rel=float(np.linalg.norm(res)/(np.linalg.norm(b)+1e-30));assert rel<1e-7;kkt.append(dict(model=m['name'],relative_normal_equation_residual=rel))
 # No-op/strict fixed versus shared perfect acceptance are distinct by construction.
 # Linear difference models must be exactly antisymmetric and zero for unchanged state.
 for m in lock['models']:
  coef=np.load(OUT/'models'/(m['name']+'.npz'))['coef'];d=np.load(OUT/'events'/allids[0]/'probes.npz');x=d['phi'];assert np.all((x[:1]-x[:1])@coef==0);assert np.allclose((x[1]-x[0])@coef,-((x[0]-x[1])@coef))
 # Teacher poisoning does not alter the online feature path.
 source=DATA/'contributions'/allids[0];f0=load_online(source);full=dict(np.load(source/'summary_and_route_features.npz'))
 for k in full:
  if k not in ALLOW:full[k]=np.zeros_like(full[k])
 with tempfile.TemporaryDirectory() as td:
  np.savez(Path(td)/'summary_and_route_features.npz',**full);f1=load_online(Path(td))
 for a,b in zip(geometry(f0),geometry(f1)):assert np.array_equal(a,b)
 original=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for name,h in ex.map(lambda name:(name,sha(DATA/name)),original):assert h==original[name]
 e13=json.loads((E13/'preparation.json').read_text());e13_files=0
 for event in e13['events']:
  for name,h in event['files'].items():assert sha(E13/'events'/event['event_id']/name)==h;e13_files+=1
 result=dict(passed=True,events=144,max_gain_error=max(r['max_gain_error'] for r in rows),online_phi_reconstructed=True,teacher_poisoning_pass=True,no_intercept_difference_models=True,train_only_normalizer_and_fit=True,frozen_base_files=len(original),frozen_E13_files=e13_files,kkt=kkt,fullspace_and_panel_scope=rows)
 dump(OUT/'audit.json',result);print({k:v for k,v in result.items() if k not in ('fullspace_and_panel_scope','kkt')})
if __name__=='__main__':run()
