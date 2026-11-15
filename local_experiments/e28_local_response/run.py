"""E28 analytic diagnostics only; development split, frozen C/phi/B, no test."""
from ..e27_adaptive_query.core import *
from ..e27_adaptive_query.analyze import load_phi
import hashlib, collections
DEST=ROOT/'artifacts/e28_local_response_v1'
UNIFORM_LOCAL=False

def digest(*xs):
 h=hashlib.sha256()
 for x in xs:
  x=x.detach().cpu().contiguous();h.update(str((x.shape,x.dtype)).encode());h.update(x.view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def fitmany(F,lm,Y):
 # Both targets use exactly the same mass-weighted ridge rule.
 # Y is t/m for centered target, deltaO for local target.
 w=torch.ones_like(lm) if UNIFORM_LOCAL else (lm-lm.max(0).values[None]).exp();X=(w[...,None]*F).permute(1,0,2);T=(w[...,None]*Y).permute(1,0,2)
 coef,pred,a=ridge_fit(X,T,1.0)
 return coef,torch.einsum('qpf,pfd->qpd',F,coef),a

def fitcenter(F,lm,Y):
 mx=lm.max(0).values;X=((lm-mx[None]).exp()[...,None]*F).permute(1,0,2)
 coef,pred,a=ridge_fit(X,Y.permute(1,0,2),torch.load(OUT/'fit.pt',weights_only=True)['zscale'])
 return coef,pred.permute(1,0,2),mx

def coefdirection(a,b):
 a=a/a.abs().amax((1,2),keepdim=True).clamp_min(1e-300);b=b/b.abs().amax((1,2),keepdim=True).clamp_min(1e-300)
 return nmse(a,b)

def summary_output(c,s,basis):
 nu=s.mean[0].double()[:,None]+torch.einsum('ag,pak,dk->pgd',haar(c['logit']).double(),s.detail[0].double(),basis[0].bfloat16().double())
 return torch.einsum('qpg,pgd->qpd',c['logit'].softmax(-1),nu)

def reconstruct(r,beta,lm,center):
 # exp(logmass+beta), not exp(beta) followed by tiny mass.
 return (lm+beta).exp()[...,None]*r + (torch.expm1(beta.clamp(-.5,.5))*lm.exp()).where(beta.abs()<.5,(lm+beta).exp()-lm.exp())[...,None]*center

@torch.no_grad()
def run(seed):
 torch.set_num_threads(2);dest=DEST/(f's{seed}_uniform' if UNIFORM_LOCAL else f's{seed}');dest.mkdir(exist_ok=True)
 bundle=bundle_for('C',seed);net=load_phi(seed);B=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda().double();qs=torch.linspace(0,63,16,device='cuda').round().long()
 groups=collections.defaultdict(list)
 for idx,path in enumerate(files('development')):
  # Filename metadata read only from development manifest.
  ev=torch.load(path,map_location='cpu',weights_only=True);groups[(ev['trajectory'],ev['layer'],ev['head'])].append((idx,path))
 rows=[];fits=[];identities=[];audit=[];conditions=[];start=time.time()
 for group,paths in groups.items():
  records=[];occ=collections.defaultdict(list)
  for idx,path in paths:
   d,s,c,t,beta,lm=prepare(bundle,path);F=net(d['q'],s);ev=d['ev'];kv=torch.load(path.parent/ev['kv_file'],map_location='cpu',weights_only=True)
   exact=ev['page_output'].cuda().double() if 'page_output' in ev else d['N']/d['Z'][...,None]
   approx=summary_output(c,s,bundle.layers[d['key']].basis);local=(exact-approx)@B;center=(approx-c['o'][:,None])@B
   # local targets and centered per-mass targets in stable canonical form.
   centered=t # Never form t/m or exp(beta) separately: these can overflow.
   identity=reconstruct(local,beta,lm,center)
   rel=float((identity-t).abs().max()/t.abs().max().clamp_min(1e-300))
   audit.append(dict(event=str(path.relative_to(OUT)),identity_relative_max=rel))
   assert rel<1e-7,(idx,str(path),rel)
   coefO,pO,_=fitcenter(F,lm,centered);coefL,pL,_=fitmany(F,lm,local)
   oldz=pO;localz=reconstruct(pL,beta,lm,center)
   mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);panel=np.load(E25/f'C_s{seed}'/f'panel_{idx:04d}.npz');assert np.array_equal(mask.cpu(),panel['states'][0]);assert np.array_equal(r.cpu(),panel['initial_remove']) and np.array_equal(a.cpu(),panel['initial_add'])
   rec=dict(idx=idx,path=path,event=str(path.relative_to(OUT)),ev=ev,F=F,lm=lm,beta=beta,local=local,centered=centered,center=center,t=t,hp=d['hp'],mask=mask,r=r,a=a,J=float(objective(c,d['N'],d['Z'],d['dense'],mask)),norm=d['dense'][qs].square().sum()/16+1e-12,g=torch.as_tensor(panel['true_initial_gain'],device='cuda'),panel={n:panel[n] for n in ['states','remove','add','true_gain']},pred={'Centered-Oracle64':oldz,'Local-Oracle64':localz,'Centered-Shared':oldz.clone(),'Local-Shared':localz.clone(),'Centered-LOTO':oldz.clone(),'Local-LOTO':localz.clone()},coefs={'Centered':coefO,'Local':coefL})
   ri=len(records);records.append(rec)
   # Identity includes ordered raw token IDs, actual BF16 KV and actual new summary.
   for p in range(d['hp']):
    raw=digest(kv['ids'][p],kv['k'][p],kv['v'][p]);summ=digest(*(getattr(s,n)[0,p] for n in ['keys','mean','detail','stats']));key=(raw,summ)
    occ[key].append((ri,p))
   if idx%24==0:print('PREP',seed,idx,flush=True)
  signatures=collections.defaultdict(list)
  for key,items in occ.items():
   identities.append(dict(group=list(group),raw_sha=key[0],summary_sha=key[1],occurrences=[dict(event=records[i]['event'],page=p) for i,p in items]))
   if len(items)>1:signatures[tuple(i for i,p in items)].append(items)
  for sig,items in signatures.items():
   # Batch all pages with the same occurrence pattern.
   ix=[torch.tensor([it[j][1] for it in items],device='cuda') for j in range(len(sig))]
   fs=[records[i]['F'][:,pp] for i,pp in zip(sig,ix)];ls=[records[i]['lm'][:,pp] for i,pp in zip(sig,ix)]
   for target in ['Centered','Local']:
    field='centered' if target=='Centered' else 'local';ys=[records[i][field][:,pp] for i,pp in zip(sig,ix)]
    C,_,cm=(fitcenter if target=='Centered' else fitmany)(torch.cat(fs),torch.cat(ls),torch.cat(ys))
    for j,(i,pp) in enumerate(zip(sig,ix)):
     rec=records[i];shared=torch.einsum('qpf,pfd->qpd',fs[j],C)
     if target=='Centered':shared=shared*(ls[j]-cm[None]).exp()[...,None]
     other=[k for k in range(len(sig)) if k!=j]
     CL,_,clm=(fitcenter if target=='Centered' else fitmany)(torch.cat([fs[k] for k in other]),torch.cat([ls[k] for k in other]),torch.cat([ys[k] for k in other]))
     held=torch.einsum('qpf,pfd->qpd',fs[j],CL)
     if target=='Centered':held=held*(ls[j]-clm[None]).exp()[...,None]
     for mode,yhat in [('Shared',shared),('LOTO',held)]:
      zz=yhat if target=='Centered' else reconstruct(yhat,rec['beta'][:,pp],ls[j],rec['center'][:,pp])
      rec['pred'][target+'-'+mode][:,pp]=zz
      fits.append(dict(seed=seed,event=rec['event'],target=target,mode=mode,pages=len(pp),occurrences=len(sig),target_nmse=nmse(yhat,ys[j]),residual_nmse=nmse(zz,rec['t'][:,pp]),coefficient_shift=coefdirection(C if mode=='Shared' else CL,rec['coefs'][target][pp])))
  for rec in records:
   p=rec['panel'];scoresout={};g=rec['g']
   for method,z in rec['pred'].items():
    score,ind=safe_predictions(z[qs],rec['beta'][qs],rec['lm'][qs],rec['mask'],rec['r'],rec['a'],rec['norm'])
    for policy,sc in [('Set',score),('Independent',ind)]:
     j=int(sc.argmax());ex=bool(sc[j]>0);ev=rec['ev'];rows.append(dict(seed=seed,event=rec['event'],trajectory=ev['trajectory'],layer=ev['layer'],head=ev['head'],chunk=ev['chunk'],step=ev['step'],tile=ev['tile'],method=method,policy=policy,initial=rec['J'],oracle_gain=max(0.,float(g.max())),proposal_gain=float(g[j]),hindsight_gain=max(0.,float(g[j])),executed_gain=float(g[j]) if ex else 0.,execute=ex,remove=int(rec['r'][j]),add=int(rec['a'][j]),predicted_gain=float(sc[j])))
    rr=torch.as_tensor(p['remove'],device='cuda');aa=torch.as_tensor(p['add'],device='cuda');preds=[]
    for st in p['states']:preds.append(safe_predictions(z[qs],rec['beta'][qs],rec['lm'][qs],torch.as_tensor(st,device='cuda'),rr,aa,rec['norm'])[0].cpu().numpy())
    preds=np.stack(preds);scoresout[method]=preds;conditions.append(dict(seed=seed,event=rec['event'],method=method,**conditional_metrics(p['true_gain'],preds)))
   np.savez_compressed(dest/f"panel_{rec['idx']:04d}.npz",event=rec['event'],true_gain=p['true_gain'],**scoresout)
  dump(dest/'status.json',dict(groups_done=len(set(r['trajectory']+str(r['layer'])+str(r['head']) for r in rows)),events=len(rows)//12,seconds=time.time()-start));print('GROUP',seed,group,'pages',len(occ),'repeated',sum(len(v) for v in signatures.values()),flush=True)
 dump(dest/'decisions.json',rows);dump(dest/'fits.json',fits);dump(dest/'identities.json',identities);dump(dest/'identity_audit.json',audit);dump(dest/'conditional.json',conditions);dump(dest/'complete.json',dict(seed=seed,events=len(paths)*len(groups),seconds=time.time()-start,C_sha=sha(checkpoint('C',seed)),phi_sha=sha(E27/f's{seed}'/'selected.pt'),test_accessed=False,training=False));print('DONE',seed,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--uniform-local',action='store_true');args=ap.parse_args();UNIFORM_LOCAL=args.uniform_local;run(args.seed)
