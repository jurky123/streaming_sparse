from ..e30_beta_diagnostics.run import *
from ..e25_decision_diagnostics.report import aggregate
import re
E30=DEST
DEST=ROOT/'artifacts/e34_key_cumulants_v1'
METHODS=['E29','Variance-only','Full-Mean+Variance','LowRank8+Variance','Exact-Q-Cumulant2','TrueBeta']

@torch.no_grad()
def fit_basis(seed):
 torch.set_num_threads(2);dest=DEST/f's{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'basis.pt').exists();bundle=bundle_for('C',seed);grams={};seen=collections.defaultdict(set);sources=[];kvseen=set();start=time.time()
 for path in files('train'):
  ev=torch.load(path,map_location='cpu',weights_only=True);kp=path.parent/ev['kv_file']
  if str(kp) in kvseen:continue
  kvseen.add(str(kp));d=load(path);s=bundle.make(d,'C');kv=torch.load(kp,map_location='cpu',weights_only=True);km=d['k'][0].double().reshape(-1,4,16,128).mean(2);offset=km-s.keys[0].double();key=d['key'];selected=[]
  for p in range(len(offset)):
   identity=ev['trajectory']+':'+digest(kv['ids'][p],kv['k'][p],s.keys[0,p])
   if identity not in seen[key]:seen[key].add(identity);selected.append(p)
  if selected:
   x=offset[selected].reshape(-1,128);G=x.T@x
   if key not in grams:grams[key]=G
   else:grams[key]+=G
  sources.append(dict(event=str(path.relative_to(OUT)),kv=str(kp.relative_to(OUT)),kv_sha=sha(kp),unique_pages_added=len(selected),key=key))
  if len(sources)%72==0:print('BASIS',seed,len(sources),round(time.time()-start),flush=True)
 bases={};spectra={};audit=[]
 for key,G in grams.items():
  G=G/(len(seen[key])*4);values,U=torch.linalg.eigh(G);U=U[:,-8:].flip(-1);values=values.flip(0);b=U.float();bases[key]=b.cpu();spectra[key]=values.cpu();audit.append(dict(key=key,unique_pages=len(seen[key]),energy8=float(values[:8].sum()/values.sum()),orthogonality_error=float((b.double().T@b.double()-torch.eye(8,device='cuda')).abs().max()),eigen_residual=float((G@U-U*values[:8]).abs().max())))
 torch.save(dict(bases=bases,spectra=spectra,grams={k:v.cpu() for k,v in grams.items()}),dest/'basis.pt');dump(dest/'basis_lock.json',dict(seed=seed,split='train',sources=sources,audit=audit,basis_sha=sha(dest/'basis.pt'),C_sha=sha(checkpoint('C',seed)),protocol_sha=sha(DEST/'protocol.json'),seconds=time.time()-start));print('BASIS DONE',seed,len(sources),flush=True)

def beta_from_c(logits,correction):return (logits+correction).logsumexp(-1)-logits.logsumexp(-1)

def online_estimates(q,keys,stats,coef,U):
 # Deployable beta algebra: no raw K/V, Dense output, true mass or labels.
 q=q.double();l=torch.einsum('qd,pgd->qpg',q,keys.double())/math.sqrt(128);q2=q.square().reshape(len(q),8,16).sum(-1)/128;variance=torch.einsum('qj,pgj->qpg',q2,stats[...,:8].double());low=torch.einsum('qr,pgr->qpg',q@U.double(),coef.double())/math.sqrt(128)
 return beta_from_c(l,.5*variance),beta_from_c(l,low+.5*variance),l,variance,low

def estimates(q,keys,stats,offset,coef,U,rawkeys):
 varbeta,lowbeta,l,variance,low=online_estimates(q,keys,stats,coef,U);mean=torch.einsum('qd,pgd->qpg',q.double(),offset)/math.sqrt(128)
 rawlogit=torch.einsum('qd,pgtd->qpgt',q.double(),rawkeys.double().reshape(-1,4,16,128))/math.sqrt(128);truevar=rawlogit.var(-1,unbiased=False)
 return dict(zip(METHODS[1:5],[varbeta,beta_from_c(l,mean+.5*variance),lowbeta,beta_from_c(l,mean+.5*truevar)])),dict(mean=mean,lowmean=low,variance=variance,truevar=truevar,logits=l)

def support_maps(seed):
 ident=json.loads((ROOT/f'artifacts/e28_local_response_v1/s{seed}/identities.json').read_text());out=collections.defaultdict(lambda:collections.defaultdict(set));raw={}
 def meta(e):return tuple(map(int,re.search(r'c(\d+)_l(\d+)_s(\d+)_h(\d+)_t(\d+)',e).groups()))
 for x in ident:
  occ=x['occurrences']
  for o in occ:
   c,l,s,h,t=meta(o['event']);raw[o['event'],o['page']]=(x['raw_sha'],x['summary_sha'])
   for d in occ:
    if d['event']==o['event']:continue
    dc,dl,ds,dh,dt=meta(d['event'])
    if dc==c and ds==s and dt!=t:out[o['event']]['tile'].add(o['page'])
    if dc==c and ds!=s:out[o['event']]['step'].add(o['page'])
    if dc<c:out[o['event']]['forward'].add(o['page'])
 return out,raw

@torch.no_grad()
def run(seed,limit=None):
 torch.set_num_threads(2);dest=DEST/f's{seed}';assert not (dest/'complete.json').exists();lock=json.loads((dest/'basis_lock.json').read_text());assert lock['basis_sha']==sha(dest/'basis.pt') and lock['protocol_sha']==sha(DEST/'protocol.json');bases=torch.load(dest/'basis.pt',weights_only=True)['bases'];bundle=bundle_for('C',seed);qs=torch.linspace(0,63,16,device='cuda').round().long();supports,ident=support_maps(seed);seen={};reuse=[];rows=[];quality=[];conds=[];moments=[];coverage=[];aud=[];start=time.time()
 for idx,path in enumerate(files('development')):
  if limit is not None and idx>=limit:break
  d,s,c,_,_,_,_=prepare_local(bundle,path);ev=d['ev'];event=str(path.relative_to(OUT));x=torch.load(E30/f's{seed}/beta_{idx:04d}.pt',map_location='cuda',weights_only=True);p=np.load(E30/f's{seed}/panel_{idx:04d}.npz');mask=torch.as_tensor(p['states'][0],device='cuda');r=torch.as_tensor(p['initial_remove'],device='cuda');a=torch.as_tensor(p['initial_add'],device='cuda');U=bases[d['key']].cuda();raw=d['k'][0];offset=raw.double().reshape(-1,4,16,128).mean(2)-s.keys[0].double();coef=(offset@U.double()).bfloat16();bb,mm=estimates(d['q'],s.keys[0],s.stats[0],offset,coef,U,raw);betas={'E29':x['betas']['Predicted'],**bb,'TrueBeta':x['beta_true']};P=len(mask);candidate=torch.zeros_like(mask);candidate[r]=True;candidate[a]=True;hist=torch.arange(P,device='cuda')<d['hp'];region={'all':torch.ones_like(mask),'candidate':candidate,'background':~candidate,'history':hist,'current':~hist}
  for kind,pp in supports[event].items():
   sel=torch.zeros_like(mask);sel[list(pp)]=True;region['reuse-'+kind]=sel;coverage.append(dict(seed=seed,event=event,kind=kind,pages=len(pp),history_pages=d['hp'],history_mass_fraction=float(x['logmass'][:,sel].exp().sum()/x['logmass'][:,hist].exp().sum())))
  for page in range(d['hp']):
   key=ident[event,page];reusekey=(ev['trajectory'],d['key'],*key);h=digest(coef[page],s.stats[0,page,:,:8]);previous=seen.get(reusekey)
   if previous is not None:assert previous==h
   else:seen[reusekey]=h
   reuse.append(dict(seed=seed,event=event,page=page,raw_sha=key[0],summary_sha=key[1],coef_and_variance_sha=h,repeated=previous is not None))
  g=torch.as_tensor(p['true_initial_gain'],device='cuda');J=float(objective(c,d['N'],d['Z'],d['dense'],mask));norm=d['dense'][qs].square().sum()/16+1e-12;meta=dict(seed=seed,event=event,**{k:ev[k] for k in ['trajectory','layer','head','chunk','step','tile']});pred={};ini={}
  for method,beta in betas.items():
   assert torch.isfinite(beta).all();sc,ind=local_scores(x['predicted_response'][qs],beta[qs],x['logmass'][qs],x['center'][qs],mask,r,a,norm);ini[method]=sc.cpu().numpy()
   for policy,score in [('Set',sc),('Independent',ind)]:
    j=int(score.argmax());ex=bool(score[j]>0);new=mask.clone()
    if ex:new[r[j]]=False;new[a[j]]=True
    assert int(new.sum())==int(mask.sum()) and bool(new[d['own']]);gain=float(g[j]) if ex else 0.
    rows.append(dict(**meta,method=method,policy=policy,initial=J,oracle_gain=max(0.,float(g.max())),proposal_gain=float(g[j]),hindsight_gain=max(0.,float(g[j])),executed_gain=gain,execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(score[j]),regret=max(0.,float(g.max()))-gain,final_nmse=J-gain,absolute_nmse_increase=-gain))
   for qname,ii in [('M16',qs),('Q64',torch.arange(64,device='cuda'))]:
    for name,sel in region.items():
     if not sel.any():continue
     error=beta[ii][:,sel]-x['beta_true'][ii][:,sel];quality.append(dict(**meta,method=method,queries=qname,region=name,pages=int(sel.sum()),beta_bias=float(error.mean()),**weighted_metrics(beta[ii],x['beta_true'][ii],x['logmass'][ii],sel)))
   scores=np.stack([local_scores(x['predicted_response'][qs],beta[qs],x['logmass'][qs],x['center'][qs],torch.as_tensor(st,device='cuda'),torch.as_tensor(p['remove'],device='cuda'),torch.as_tensor(p['add'],device='cuda'),norm)[0].cpu().numpy() for st in p['states']]);pred[method]=scores;conds.append(dict(**meta,method=method,**conditional_metrics(p['true_gain'],scores)))
  assert np.allclose(ini['E29'],p['initial_Predicted'],rtol=1e-9,atol=1e-10) and np.allclose(ini['TrueBeta'],p['initial_True'],rtol=1e-9,atol=1e-10)
  moments.append(dict(**meta,mean_rms=float(mm['mean'].square().mean().sqrt()),low_mean_error_rms=float((mm['lowmean']-mm['mean']).square().mean().sqrt()),mean_projection_relative_mse=float((mm['lowmean']-mm['mean']).square().sum()/mm['mean'].square().sum().clamp_min(1e-300)),variance_approx_bias=float((mm['variance']-mm['truevar']).mean()),variance_relative_mse=float((mm['variance']-mm['truevar']).square().sum()/mm['truevar'].square().sum()),full_c2_beta_bias=float((betas['Exact-Q-Cumulant2']-x['beta_true']).mean())))
  torch.save(dict(event=event,coefficients_bf16=coef.cpu(),beta={k:v.cpu() for k,v in betas.items()},basis_key=d['key']),dest/f'beta_{idx:04d}.pt');np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=event,states=p['states'],remove=p['remove'],add=p['add'],initial_remove=p['initial_remove'],initial_add=p['initial_add'],true_gain=p['true_gain'],true_initial_gain=p['true_initial_gain'],**pred,**{'initial_'+k:v for k,v in ini.items()});aud.append(dict(**meta,original_mask_unchanged=True,all_beta_finite=True))
  if (idx+1)%24==0:
   for name,data in [('decisions',rows),('quality',quality),('conditional',conds),('moments',moments),('coverage',coverage),('reuse',reuse),('audit',aud)]:
    for row in data:
     for k,v in row.items():
      if isinstance(v,float) and not math.isfinite(v):row[k]=None
    dump(dest/(name+'.json'),data)
   print('EVENTS',seed,idx+1,round(time.time()-start),flush=True)
 for name,data in [('decisions',rows),('quality',quality),('conditional',conds),('moments',moments),('coverage',coverage),('reuse',reuse),('audit',aud)]:dump(dest/(name+'.json'),data)
 if limit is None:dump(dest/'complete.json',dict(seed=seed,events=288,seconds=time.time()-start,basis_sha=sha(dest/'basis.pt'),reader_sha=sha(E29/f's{seed}/selected.pt'),C_sha=sha(checkpoint('C',seed)),protocol_sha=sha(DEST/'protocol.json'),network_training=False,test_accessed=False))
 print('DONE',seed,len(aud),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--basis',action='store_true');p.add_argument('--limit',type=int);a=p.parse_args();fit_basis(a.seed) if a.basis else run(a.seed,a.limit)
