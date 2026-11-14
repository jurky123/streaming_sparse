import concurrent.futures
from .core import *

def one(eid):
 out=OUT/'events'/eid
 if (out/'identity.json').exists():return eid
 out.mkdir(parents=True,exist_ok=True);root=DATA/'contributions'/eid;online=load_online(root);p=pages(online);raw=dict(np.load(root/'residuals_fp64.npz'));own=int(raw['own']);K=int(raw['K']);risk=initial_mask(p,own,K);assert np.array_equal(risk,raw['risk'])
 with np.load(DATA/'searches/v1'/eid/'candidate_gain_surfaces.npz') as z:trajectory=z['state_masks'];assert np.array_equal(trajectory[0],risk)
 candidates=[('risk',risk)]+[(f'oracle_step{k}',trajectory[min(k,len(trajectory)-1)]) for k in (1,2,4)]
 rng=np.random.default_rng(int.from_bytes(hashlib.sha256((eid+'E13-seed17').encode()).digest()[:8],'little'))
 for k in (1,2,4):
  r=rng.choice(np.flatnonzero(risk & (np.arange(len(risk))!=own)),k,replace=False);a=rng.choice(np.flatnonzero(~risk),k,replace=False);x=risk.copy();x[r]=False;x[a]=True;candidates.append((f'uniform_perturb{k}',x))
 states=[];names=[];aliases=[];keys={}
 for name,x in candidates:
  key=x.tobytes()
  if key in keys:aliases[keys[key]].append(name)
  else:keys[key]=len(states);states.append(x.copy());names.append(name);aliases.append([name])
 states=np.stack(states);assert np.all(states[:,own]) and np.all(states.sum(1)==K)
 feats=[];gains=[];rs=[];aa=[];js=[];checkerr=0.
 for x in states:
  r,a=pool(p,x,own);F=pair_features(p,x,r,a);g=swap_labels(raw,x,r,a);j=objective(raw,x)
  # Separate full-mask evaluation for selected actions; not incremental swap algebra.
  for idx in (0,17,255,511):
   y=x.copy();y[r[idx]]=False;y[a[idx]]=True;err=abs(j-objective(raw,y)-g[idx]);assert err<1e-10;checkerr=max(checkerr,err)
  feats.append(F);gains.append(g);rs.append(r);aa.append(a);js.append(j)
 commonr=np.flatnonzero(states.all(0));commonr=commonr[commonr!=own];commona=np.flatnonzero((~states).all(0));r=commonr[np.argsort(p['risk'][commonr],kind='stable')[:16]];a=commona[np.argsort(-p['risk'][commona],kind='stable')[:32]];r=np.repeat(r,len(a));a=np.tile(a,len(r)//len(a));assert len(r)==len(a)>0
 anchorF=np.stack([pair_features(p,x,r,a) for x in states]);anchorG=np.stack([swap_labels(raw,x,r,a) for x in states]);assert np.all(anchorF[:,:,:20]==anchorF[0:1,:,:20])
 np.savez_compressed(out/'states.npz',features=np.stack(feats),gains=np.stack(gains),remove=np.stack(rs),add=np.stack(aa),masks=states,nmse=np.array(js),own=own,K=K,names=np.array(names))
 np.savez_compressed(out/'anchors.npz',features=anchorF,gains=anchorG,remove=r,add=a)
 dump(out/'identity.json',dict(event_id=eid,split=split(eid),states=names,aliases=aliases,count=len(states),anchor_pairs=len(r),max_direct_error=checkerr,files={f:sha(out/f) for f in ('states.npz','anchors.npz')},source_identity_sha256=sha(root/'identity.json')));return eid
if __name__=='__main__':
 preregister();ids=sorted(p.name for p in (DATA/'contributions').iterdir())
 with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:
  for i,eid in enumerate(ex.map(one,ids),1):print(i,eid,flush=True)
 metas=[json.loads((OUT/'events'/eid/'identity.json').read_text()) for eid in ids];dump(OUT/'preparation.json',dict(event_count=len(ids),states=sum(m['count'] for m in metas),by_split={sp:sum(m['count'] for m in metas if m['split']==sp) for sp in ('train','validation','evaluation')},max_direct_error=max(m['max_direct_error'] for m in metas),events=metas))
