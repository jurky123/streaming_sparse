import concurrent.futures
from .core import *

def value_panel(G,J):
 S,A=G.shape;best=np.maximum(G.max(1),0);best_ids=np.where(G.max(1)>1e-12,G.argmax(1),-1);jmean=int(G.mean(0).argmax());jplus=int(np.maximum(G,0).mean(0).argmax());fixed=G[:,jmean];fixedplus=G[:,jplus];clip=np.maximum(fixed,0);strong=np.maximum(fixedplus,0);strict=max(0,float(fixed.mean()));V=best-clip;Vs=best-strong
 # Best choices can tie; material regret handles these ties without declaring useful change.
 j0=best_ids[0];g0_action=np.zeros(S) if j0<0 else G[:,j0];rates=[];tri=np.triu(np.ones((A,A),bool),1);base=G[0,:,None]-G[0,None,:]
 for threshold in [.001,.01,.05]:
  eps=np.maximum(1e-12,threshold*J);material_sign=((G>eps[:,None]).any(0)&(G< -eps[:,None]).any(0)).mean();num=den=matden=0
  for s in range(1,S):
   d=G[s,:,None]-G[s,None,:];valid=tri&(abs(base)>1e-12)&(abs(d)>1e-12);material=valid&(abs(base)>=eps[0])&(abs(d)>=eps[s]);flip=material&(base*d<0);num+=int(flip.sum());den+=int(valid.sum());matden+=int(material.sum())
  rates.append(dict(relative_threshold=threshold,material_sign_flip_fraction=float(material_sign),material_ranking_flips_over_nonties=num/den if den else None,ranking_flips_among_material_pairs=num/matden if matden else None,material_best_action_change_fraction=float(np.mean((best[1:]-g0_action[1:])>=eps[1:])),material_conditional_advantage_fraction=float(np.mean(V>=eps))))
 # Leave-one-state-out fixed preference is a separate descriptive robustness panel.
 loso=[]
 for i in range(S):
  means=(G.sum(0)-G[i])/(S-1);j=int(means.argmax());loso.append(max(0,float(G[i,j])))
 return dict(states=S,actions=A,j_mean=jmean,j_positive=jplus,conditional_gain=float(best.mean()),fixed_mean_gain=float(fixed.mean()),strict_fixed_gain=strict,fixed_with_perfect_accept_gain=float(clip.mean()),strong_fixed_with_perfect_accept_gain=float(strong.mean()),V_requested=float(V.mean()),V_strict=float(best.mean()-strict),V_strong_fixed=float(Vs.mean()),V_over_mean_state_nmse=float(V.mean()/J.mean()),V_statewise_relative=float(np.mean(V/J)),V_fraction_conditional_gain=float(V.mean()/best.mean()) if best.mean()>1e-12 else None,V_strong_fraction_conditional_gain=float(Vs.mean()/best.mean()) if best.mean()>1e-12 else None,strict_acceptance_extra=float(clip.mean()-strict),best_action_change_vs_risk=float(np.mean(best_ids[1:]!=best_ids[0])),distinct_best_actions=len(np.unique(best_ids)),LOSO_fixed_with_perfect_accept_gain=float(np.mean(loso)),thresholds=rates,state_details=[dict(state=i,nmse=float(J[i]),conditional_gain=float(best[i]),fixed_gain=float(fixed[i]),fixed_with_perfect_accept_gain=float(clip[i]),strong_fixed_with_perfect_accept_gain=float(strong[i]),best_action=int(best_ids[i]),j_mean=jmean,j_positive=jplus) for i in range(S)])

def one(eid):
 out=OUT/'events'/eid
 if (out/'result.json').exists():return eid
 out.mkdir(parents=True,exist_ok=True);an=np.load(E13/'events'/eid/'anchors.npz');st=np.load(E13/'events'/eid/'states.npz');G=an['gains'];J=st['nmse'];r=an['remove'];a=an['add'];masks=st['masks'];f=load_online(DATA/'contributions'/eid);mass,h,U=geometry(f);proxy=mass[...,None]*h;H=np.stack([proxy[:,x].sum(1) for x in masks]);v=proxy[:,r]-proxy[:,a]
 product=-np.einsum('sqi,qaj->saij',H,v)/len(mass);sc=an['features'][:,:,20:24];context=an['features'][:,:,20:32];dot=np.trace(product,axis1=-2,axis2=-1)[...,None];phi=np.concatenate((context,np.concatenate((sc,dot),-1),np.concatenate((sc,product.reshape(len(G),len(r),64)),-1)),-1);assert phi.shape[-1]==85
 raw=np.load(DATA/'contributions'/eid/'residuals_fp64.npz');qi=f['probe_indices'].astype(int);ee=raw['e'][qi];trueE=np.stack([ee[:,~x].sum(1) for x in masks]);trueD=ee[:,r]-ee[:,a];EP=trueE@U;DP=trueD@U;Q=np.linalg.qr(U)[0]
 sh,shvalid=cosine(H,EP);ag,agvalid=cosine(H.mean(1),EP.mean(1));dc,dcvalid=cosine(v,DP);dg,dgvalid=cosine(v.mean(0),DP.mean(0));unweighted,uv=cosine(h[:,r]-h[:,a],DP)
 rng=np.random.default_rng(17);nullS=[];nullD=[]
 for _ in range(16):
  rotation=np.linalg.qr(rng.normal(size=(8,8)))[0];cs,vs=cosine(H,EP@rotation);cd,vd=cosine(v,DP@rotation);nullS.append(float(np.mean(np.abs(cs[vs]))));nullD.append(float(np.mean(np.abs(cd[vd]))))
 energyE=np.square(trueE).sum();energyD=np.square(trueD).sum();direction=dict(set_signed_cosine=float(sh[shvalid].mean()),negative_set_signed_cosine=float(-sh[shvalid].mean()),set_abs_cosine=float(np.abs(sh[shvalid]).mean()),set_aggregate_signed_cosine=float(ag[agvalid].mean()),set_random_rotation_abs_cosine=float(np.mean(nullS)),candidate_signed_cosine=float(dc[dcvalid].mean()),candidate_abs_cosine=float(np.abs(dc[dcvalid]).mean()),candidate_aggregate_signed_cosine=float(dg[dgvalid].mean()),candidate_random_rotation_abs_cosine=float(np.mean(nullD)),candidate_unweighted_signed_cosine=float(unweighted[uv].mean()),true_set_retained_energy=float(np.square(trueE@Q).sum()/energyE) if energyE>0 else None,true_candidate_retained_energy=float(np.square(trueD@Q).sum()/energyD) if energyD>0 else None,basis_condition=float(np.linalg.cond(U)),valid_set_queries=int(shvalid.sum()),valid_candidate_queries=int(dcvalid.sum()))
 # Rank8 oracle bilinear interaction versus its online proxy, Q16 only; no pooling before products.
 oracle_dot=np.einsum('sqd,qad->sa',EP,DP)/len(mass);online_dot=dot[...,0]
 np.savez_compressed(out/'probes.npz',phi=phi,pair=an['features'][0,:,:20],gains=G,nmse=J,scale=max(float(G.std()),1e-12),set_cosine=sh,candidate_cosine=dc,online_dot=online_dot,oracle_dot=oracle_dot,H=H,E_projected=EP,remove=r,add=a,state_names=st['names'])
 result=dict(event_id=eid,split=split(eid),A=value_panel(G,J),B_direction=direction,source_anchor_sha256=sha(E13/'events'/eid/'anchors.npz'),source_state_sha256=sha(E13/'events'/eid/'states.npz'),file_sha256=sha(out/'probes.npz'));dump(out/'result.json',result);return eid
if __name__=='__main__':
 preregister();ids=sorted(p.name for p in (E13/'events').iterdir())
 with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:
  for i,eid in enumerate(ex.map(one,ids),1):print(i,eid,flush=True)
 dump(OUT/'preparation.json',dict(events=144,states=994,event_ids=ids,complete=True))
