from .core import *
def one(eid):
 dest=OUT/'events'/eid;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'READY.json').exists():return eid
 st=np.load(E13/'events'/eid/'states.npz');an=np.load(E13/'events'/eid/'anchors.npz');raw=np.load(DATA/'contributions'/eid/'residuals_fp64.npz');f=load_online(DATA/'contributions'/eid)
 G=an['gains'];r=an['remove'];a=an['add'];masks=st['masks'];qi=f['probe_indices'].astype(int);e=raw['e'];de=e[:,r]-e[:,a];de2=np.square(de).sum(-1);C=np.square(raw['dense_output']).sum()+1e-12
 assert np.all(masks[:,r]) and not np.any(masks[:,a]);S,A=G.shape;I=[];B=[];D=[];I16=[];B16=[];D16=[];zs=[];znews=[]
 for x in masks:
  E=e[:,~x].sum(1);z=np.where(x[None],raw['exact_mass'],raw['summary_mass']).sum(1);zn=z[:,None]-raw['d'][:,r]+raw['d'][:,a];assert (zn>0).all()
  cross=np.einsum('qd,qad->qa',E,de);it=-2*cross/z[:,None]**2/C;bt=de2/z[:,None]**2/C
  # Direct positive norm avoids cancellation in denominator term.
  n=np.square(E[:,None]+de).sum(-1);dt=n*((zn-z[:,None])*(zn+z[:,None])/(z[:,None]**2*zn**2))/C
  I.append(it.sum(0));B.append(bt.sum(0));D.append(dt.sum(0));I16.append(it[qi].sum(0));B16.append(bt[qi].sum(0));D16.append(dt[qi].sum(0));zs.append(z[qi]);znews.append(zn[qi])
 I,B,D=map(np.array,(I,B,D));err=float(np.max(abs(I-B+D-G)));assert err<1e-10
 arrays=dict(gains=G,nmse=st['nmse'],names=st['names'],remove=r,add=a,I=I,B=B,D=D,I16=np.array(I16),B16=np.array(B16),D16=np.array(D16))
 ps,U=proxies(f);z=np.array(zs);zn=np.array(znews);checks={}
 old=np.load(E15/'events'/eid/'scores.npz')
 for rep in REPS:
  hp=ps[rep];H=np.stack([hp[:,x].sum(1) for x in masks]);v=hp[:,r]-hp[:,a];ip=2*np.einsum('sqd,qad->sqa',H,v);bp=np.square(v).sum(-1);bp=np.broadcast_to(bp[None],ip.shape)
  arrays[rep+'_I']=ip.mean(1);arrays[rep+'_B']=bp.mean(1)
  arrays[rep+'_current_z']=((ip-bp)/z[:,:,None]**2).mean(1)
  h2=np.square(H).sum(-1)[:,:,None];newnorm=np.stack([np.square(-hh[:,None]+v).sum(-1) for hh in H])
  arrays[rep+'_full_z']=(h2/z[:,:,None]**2-newnorm/zn**2).mean(1)
  checks[rep]=float(np.max(abs(arrays[rep+'_I']-arrays[rep+'_B']-old[rep])));assert checks[rep]<1e-10
  # Auxiliary norm-scale diagnostics only: true vs proxy action energy on same probes, same true denominator.
  arrays[rep+'_B_oracle_units']=(bp/z[:,:,None]**2).sum(1)/C
 np.savez_compressed(dest/'terms.npz',**arrays)
 dump(dest/'READY.json',dict(event_id=eid,split=split(eid),max_identity_error=err,e15_score_error=checks,states=S,actions=A,terms_sha=sha(dest/'terms.npz'),sources={str(p.relative_to(ROOT)):sha(p) for p in [E13/'events'/eid/'anchors.npz',E13/'events'/eid/'states.npz',DATA/'contributions'/eid/'residuals_fp64.npz',DATA/'contributions'/eid/'summary_and_route_features.npz']}));return eid
if __name__=='__main__':
 preregister()
 with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
  for i,e in enumerate(ex.map(one,ids()),1):print(i,e,flush=True)
