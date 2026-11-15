from .core import *
def retention(x,U):
 y=x@U;n=np.square(x).sum(-1);ny=np.square(y).sum(-1);valid=n>1e-25;ratio=np.divide(ny,n,out=np.zeros_like(n),where=valid);assert ratio.max()<1+1e-10
 return dict(energy_retention=float(ny.sum()/n.sum()) if n.sum()>1e-30 else None,mean_projection_cosine=float(np.sqrt(np.minimum(ratio[valid],1)).mean()) if valid.any() else None)
def one(eid):
 dest=OUT/'capacity'/eid;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return eid
 raw,f,cs=components(eid);st=np.load(E13/'events'/eid/'states.npz');an=np.load(E13/'events'/eid/'anchors.npz');old=np.load(E17/'events'/eid/'terms.npz');pca=np.load(OUT/'pca16.npz');models={'current_rank8':np.linalg.qr(f['basis'].astype(float))[0],**{n+'_PCA16':pca[n] for n in ['net','V','Z']},'full128':np.eye(128)};masks=st['masks'];r=an['remove'];a=an['add'];G=an['gains'];J=st['nmse'];S=len(G);C=np.square(raw['dense_output']).sum()+1e-12;qi=f['probe_indices'].astype(int);ident=dict(event_id=eid,split=split(eid));energy=[];perf=[];sign=[];saved={}
 z=np.stack([np.where(x[None],raw['exact_mass'],raw['summary_mass']).sum(1) for x in masks]);zn=z[:,:,None]-raw['d'][:,r][None]+raw['d'][:,a][None];assert (zn>0).all();max_full=0
 for name,U in models.items():
  for target,arr in cs.items():
   for scope,sel in [('all_pages',np.ones(arr.shape[1],bool)),('risk_unselected',~masks[0])]:energy.append(ident|dict(space=name,target=target,scope=scope,**retention(arr[:,sel],U)))
   sets=np.stack([arr[:,~x].sum(1) for x in masks]);energy.append(ident|dict(space=name,target=target,scope='risk_set',**retention(sets[0],U)));energy.append(ident|dict(space=name,target=target,scope='post_risk_sets',**retention(sets[1:],U)))
  e=cs['net']@U;E=np.stack([e[:,~x].sum(1) for x in masks]);de=e[:,r]-e[:,a];n0=np.square(E).sum(-1);gain=[];inter=[]
  for i in range(S):
   n1=np.square(E[i,:,None]+de).sum(-1);gain.append((n0[i,:,None]/z[i,:,None]**2-n1/zn[i]**2)/C);inter.append(-2*np.einsum('qd,qad->qa',E[i],de)/z[i,:,None]**2/C)
  gain=np.stack(gain);inter=np.stack(inter)
  for M,qs,factor in [(64,np.arange(64),1),(16,qi,4)]:
   score=gain[:,qs].sum(1)*factor;ip=inter[:,qs].sum(1)*factor;saved[name+'_Q'+str(M)]=score
   if name=='full128' and M==64:max_full=float(np.max(abs(score-G)));assert max_full<1e-10
   for scope,sel in [('risk',np.array([0])),('post_risk',np.arange(1,S)),('all',np.arange(S))]:
    perf.append(ident|dict(space=name,probes=M,scope=scope,**metrics(score[sel],G[sel],J[sel])))
    truth=old['I'][sel];valid=abs(truth)>1e-12;matched=old['I' if M==64 else 'I16'][sel];vm=abs(matched)>1e-12;sign.append(ident|dict(space=name,probes=M,scope=scope,I_sign_agreement_Q64=float(np.mean((ip[sel][valid]>0)==(truth[valid]>0))) if valid.any() else None,I_sign_agreement_matchingQ=float(np.mean((ip[sel][vm]>0)==(matched[vm]>0))) if vm.any() else None))
 np.savez_compressed(dest/'scores.npz',**saved)
 dump(dest/'result.json',dict(energy=energy,performance=perf,sign=sign,audit=dict(full128_max_gain_error=max_full,pca_sha=sha(OUT/'pca16.npz'),sources={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'contributions'/eid/'residuals_fp64.npz',DATA/'contributions'/eid/'summary_and_route_features.npz',E13/'events'/eid/'states.npz',E13/'events'/eid/'anchors.npz']})));return eid
if __name__=='__main__':
 with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
  for i,eid in enumerate(ex.map(one,ids()),1):print(i,eid,flush=True)
 results=[json.loads((OUT/'capacity'/eid/'result.json').read_text()) for eid in ids()]
 for name,key,keys in [('A_energy','energy',['split','space','target','scope']),('A_performance','performance',['split','space','probes','scope']),('A_interaction_sign','sign',['split','space','probes','scope'])]:
  rows=[x for r in results for x in r[key]];table(name+'_per_event.csv',rows);table(name+'_summary.csv',aggregate(rows,keys))
 dump(OUT/'A_complete.json',dict(events=144,states=994,max_full128_error=max(r['audit']['full128_max_gain_error'] for r in results)))
