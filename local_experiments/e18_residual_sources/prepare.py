from .core import *

def energy(v,z,weight=None):
 vv=np.square(v).sum(-1);zz=np.square(z).sum(-1);cross=2*np.sum(v*z,-1);ee=np.square(v+z).sum(-1)
 if weight is not None:vv=vv*weight;zz=zz*weight;cross=cross*weight;ee=ee*weight
 a,b,c,t=map(float,(vv.sum(),zz.sum(),cross.sum(),ee.sum()));assert abs(a+b+c-t)<max(1e-10,abs(t)*1e-10)
 return dict(V_energy=a,Z_energy=b,cross=c,total=t,V_over_total=a/t if t>1e-30 else None,Z_over_total=b/t if t>1e-30 else None,cross_over_total=c/t if t>1e-30 else None,Z_share_component_energy=b/(a+b) if a+b>1e-30 else None,cancellation_ratio=t/(a+b) if a+b>1e-30 else None)
def cosrow(proxy,target):
 c,valid=cosine(proxy,target)
 return dict(cosine=float(c[valid].mean()) if valid.any() else None,abs_cosine=float(abs(c[valid]).mean()) if valid.any() else None,valid=int(valid.sum()))
def one(eid):
 dest=OUT/'events'/eid;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return eid
 raw=np.load(DATA/'contributions'/eid/'residuals_fp64.npz');st=np.load(E13/'events'/eid/'states.npz');an=np.load(E13/'events'/eid/'anchors.npz');old=np.load(E17/'events'/eid/'terms.npz');f=load_online(DATA/'contributions'/eid)
 op,mass=summary_output(f);e=raw['e'];ez=raw['d'][...,None]*(op-raw['dense_output'][:,None]);ev=e-ez;qi=f['probe_indices'].astype(int);C=np.square(raw['dense_output']).sum()+1e-12;r=an['remove'];a=an['add'];G=an['gains'];masks=st['masks'];S,A=G.shape
 identity_error=float(np.max(abs(e.sum(1)-np.sum(raw['summary_mass'][...,None]*(raw['dense_output'][:,None]-op),1))));assert identity_error<1e-10
 assert np.allclose(e,ev+ez,rtol=1e-12,atol=1e-12)
 ident=dict(event_id=eid,split=split(eid));energies=[];directions=[];interactions=[];selected=[];arrays=dict(gains=G,nmse=st['nmse'],names=st['names'],remove=r,add=a)
 groups={'all_pages':np.ones(e.shape[1],bool),'risk_unselected':~masks[0]}
 for name,js in [('risk_beneficial_swap_pages',np.flatnonzero(G[0]>1e-12)),('risk_best_swap_pages',np.array([int(G[0].argmax())]) if G[0].max()>1e-12 else np.array([],int))]:
  mask=np.zeros(e.shape[1],bool);mask[np.unique(np.r_[r[js],a[js]])]=True;groups[name]=mask
 for name,mask in groups.items():
  if mask.any():energies.append(ident|dict(level='page',scope=name,state='risk',pages=int(mask.sum()),**energy(ev[:,mask],ez[:,mask])))
 dv=ev[:,r]-ev[:,a];dz=ez[:,r]-ez[:,a];Iparts=[];Iparts16=[]
 full_output=np.einsum('qp,qpd->qd',mass,op);full_hp=mass[...,None]*(op-full_output[:,None]);proxy_full=full_hp[qi];candidate_proxy=proxy_full[:,r]-proxy_full[:,a]
 for part,target in [('total',e),('V',ev),('Z',ez)]:
  for name,mask in groups.items():
   if mask.any():directions.append(ident|dict(level='page',scope=name,state='risk',component=part,**cosrow(proxy_full[:,mask],target[qi][:,mask])))
 for si,x in enumerate(masks):
  EV=ev[:,~x].sum(1);EZ=ez[:,~x].sum(1);z=np.where(x[None],raw['exact_mass'],raw['summary_mass']).sum(1);weight=1/z**2/C
  energies.append(ident|dict(level='set',scope='risk' if si==0 else 'post_risk',state=str(st['names'][si]),pages=int((~x).sum()),**energy(EV,EZ,weight)))
  terms=np.stack([-2*np.einsum('qd,qad->qa',L,R)*weight[:,None] for L,R in [(EV,dv),(EV,dz),(EZ,dv),(EZ,dz)]],-1)
  totals=terms.sum(0);Iparts.append(totals);Iparts16.append(terms[qi].sum(0));assert np.allclose(totals.sum(-1),old['I'][si],rtol=1e-9,atol=1e-11)
  H=proxy_full[:,x].sum(1)
  for name,Epart,dpart in [('total',EV+EZ,dv+dz),('V',EV,dv),('Z',EZ,dz)]:
   directions.append(ident|dict(level='set',scope='risk' if si==0 else 'post_risk',state=str(st['names'][si]),component=name,**cosrow(-H,Epart[qi])))
   directions.append(ident|dict(level='candidate',scope='risk' if si==0 else 'post_risk',state=str(st['names'][si]),component=name,**cosrow(candidate_proxy,dpart[qi])))
  picked={'oracle_best':int(G[si].argmax()),'E15_full_v':int((old['full_v_I'][si]-old['full_v_B'][si]).argmax()),'E15_mean_only':int((old['mean_only_I'][si]-old['mean_only_B'][si]).argmax())}
  masks_j={'all_candidates':np.ones(A,bool),'positive_gain':G[si]>1e-12,'negative_gain':G[si]< -1e-12}
  for typ,j in picked.items():
   sel=np.zeros(A,bool);sel[j]=True;masks_j[typ+('_positive' if G[si,j]>1e-12 else '_negative' if G[si,j]< -1e-12 else '_tie')]=sel
   selected.append(ident|dict(state=str(st['names'][si]),state_index=si,selection=typ,action=j,I_VV=float(totals[j,0]),I_VZ=float(totals[j,1]),I_ZV=float(totals[j,2]),I_ZZ=float(totals[j,3]),I=float(totals[j].sum()),B=float(old['B'][si,j]),D=float(old['D'][si,j]),true_gain=float(G[si,j])))
  for group,keep in masks_j.items():
   if not keep.any():continue
   t=totals[keep];it=t.sum(-1);ab=np.abs(t).sum(-1);valid=abs(it)>1e-12
   row=ident|dict(scope='risk' if si==0 else 'post_risk',state=str(st['names'][si]),candidates=group,count=int(keep.sum()),I_mean=float(it.mean()),V_only_sign_agreement=float(np.mean((t[valid,0]>0)==(it[valid]>0))) if valid.any() else None,Z_only_sign_agreement=float(np.mean((t[valid,3]>0)==(it[valid]>0))) if valid.any() else None)
   for k,n in enumerate(['VV','VZ','ZV','ZZ']):row['I_'+n+'_mean']=float(t[:,k].mean());row['I_'+n+'_absolute_share']=float(np.mean(abs(t[:,k])/(ab+1e-30)))
   interactions.append(row)
 arrays['I_parts']=np.array(Iparts);arrays['I_parts16']=np.array(Iparts16)
 arrays['I']=old['I'];arrays['B']=old['B'];arrays['D']=old['D'];arrays['I16']=old['I16'];arrays['B16']=old['B16']
 # Unit conversion uses ONLY online summary output energy, not true C or NMSE.
 for rep in REPS:
  outq=full_output[qi] if rep=='full_v' else np.einsum('qp,pd->qd',mass[qi],f['summary_mean'].astype(float))
  Chat=max(4*float(np.square(outq).sum()),1e-12);arrays[rep+'_unit_I']=64*old[rep+'_I']/Chat;arrays[rep+'_unit_B']=64*old[rep+'_B']/Chat;arrays[rep+'_Chat']=np.array(Chat)
 np.savez_compressed(dest/'terms.npz',**arrays)
 result=ident|dict(energy=energies,directions=directions,interactions=interactions,selected=selected,audit=dict(summary_global_identity_error=identity_error,I_decomposition_error=float(np.max(abs(arrays['I_parts'].sum(-1)-old['I']))),set_energy_error=max(abs(t['total']-float(st['nmse'][i])) for i,t in enumerate([x for x in energies if x['level']=='set'])),terms_sha=sha(dest/'terms.npz'),sources={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'contributions'/eid/'residuals_fp64.npz',DATA/'contributions'/eid/'summary_and_route_features.npz',E13/'events'/eid/'states.npz',E13/'events'/eid/'anchors.npz',E17/'events'/eid/'terms.npz']}))
 assert result['audit']['set_energy_error']<1e-10
 dump(dest/'result.json',result);return eid
if __name__=='__main__':
 preregister()
 with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
  for i,e in enumerate(ex.map(one,ids()),1):print(i,e,flush=True)
