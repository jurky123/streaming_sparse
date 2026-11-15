from .run import *

@torch.no_grad()
def main():
 torch.set_num_threads(2);paths=files('development');checks=[];counts=[];provenance=[];fixed=[];Bchecks=[]
 for seed in [17,29]:
  dest=DEST/f's{seed}';rows=json.loads((dest/'rows.json').read_text());bundle=bundle_for('C',seed);ident=json.loads((ROOT/f'artifacts/e28_local_response_v1/s{seed}/identities.json').read_text());identmap={}
  for x in ident:
   if x['group'][2]==0:
    for o in x['occurrences']:identmap[o['event'],o['page']]=(x['raw_sha'],x['summary_sha'])
  byevent=collections.defaultdict(list)
  for r in rows:byevent[r['event']].append(r)
  emeta={};edata={};evindex={}
  for idx,path in enumerate(paths):
   event=str(path.relative_to(OUT))
   if event not in byevent:continue
   ev=torch.load(path,map_location='cpu',weights_only=True);emeta[event]=ev;evindex[event]=idx;edata[event]=torch.load(dest/f'event_{idx:04d}.pt',weights_only=True)
   kv=torch.load(path.parent/ev['kv_file'],map_location='cpu',weights_only=True)
   # Recheck the raw physical identity rather than trusting the old manifest alone.
   for p in range(kv['history_pages']):assert digest(kv['ids'][p],kv['k'][p],kv['v'][p])==identmap[event,p][0]
  for event,cache in edata.items():
   ev=emeta[event];base=cache['variants']['Baseline'];P=len(base['mask']);hp=P-72
   for x in cache['provenance']:
    assert event not in x['donor_events']
    for donor,pp in zip(x['donor_events'],x['donor_pages']):
     de=emeta[donor];assert de['trajectory']==ev['trajectory'] and de['layer']==ev['layer'] and de['head']==ev['head']
     if x['kind']=='tile':assert de['chunk']==ev['chunk'] and de['step']==ev['step'] and de['tile']!=ev['tile']
     if x['kind']=='step':assert de['chunk']==ev['chunk'] and de['step']!=ev['step']
     if x['kind']=='forward':assert de['chunk']<ev['chunk']
     for p,pd in zip(x['pages'],pp):assert identmap[event,p]==identmap[donor,pd]
    provenance.append(dict(seed=seed,event=event,kind=x['kind'],pages=len(x['pages']),donors=len(x['donor_events']),target_excluded=True,raw_identity_equal=True))
   for name,v in cache['variants'].items():
    keys,lp,mask=v['keys'],v['logpi'],v['mask'];assert torch.isfinite(keys).all() and torch.isfinite(lp).all() and (lp.exp()>0).all();assert (lp.exp().sum(-1)-1).abs().max()<1e-12;assert torch.equal(keys,keys.bfloat16().double());assert int(mask.sum())==int(P*.1) and mask[ev['own']];assert torch.all(mask[v['remove']]) and torch.all(~mask[v['add']]) and not torch.any(v['remove']==ev['own'])
    if name!='Baseline':
     support=v['support'];assert not support[hp:].any();assert torch.equal(keys[~support],base['keys'][~support]) and torch.equal(lp[~support],base['logpi'][~support])
    if name.startswith('A1'):assert torch.equal(keys,base['keys'])
    if name.startswith('A2'):assert torch.equal(lp,base['logpi'])
   counts.append(dict(seed=seed,event=event,variants=len(cache['variants']),count_own_positive_weights_valid=True,unsupported_and_current_unchanged=True))
  # All fixed-mask source decompositions, before any Risk reselection.
  for event,cache in edata.items():
   idx=evindex[event];d,s,c,_,_,_,_=prepare_local(bundle,paths[idx]);m=bundle.layers[d['key']];oldmask=cache['variants']['Baseline']['mask'].cuda();sample=(idx in sorted(evindex.values())[::12])
   for name,v in cache['variants'].items():
    cc=weighted_contributions(d,s,m,v['keys'].cuda(),v['logpi'].cuda());de=decomposition(d,cc,oldmask);fixed.append(dict(seed=seed,event=event,method=name,**de));row=next(r for r in byevent[event] if r['method']==name);assert abs(de['net_nmse']-row['same_mask_nmse'])<1e-8
    if name=='Baseline':
     rel=float((cc['nh']-c['nh']).abs().max()/c['nh'].abs().max().clamp_min(1e-300));lz=float((cc['logzh']-c['logzh']).abs().max());assert rel<1e-10 and lz<1e-10;Bchecks.append(dict(seed=seed,event=event,baseline_N_relative_error=rel,baseline_logZ_absolute_error=lz))
    if not sample:continue
    # Separate NumPy longdouble N/Z construction and selected true outputs.
    q=d['q'].cpu().numpy().astype(np.longdouble);keys=v['keys'].numpy().astype(np.longdouble);lp=v['logpi'].numpy().astype(np.longdouble);nu=cc['nu'].cpu().numpy().astype(np.longdouble);scale=d['log_scale'].cpu().numpy().astype(np.longdouble);logits=np.einsum('qd,pgd->qpg',q,keys)/np.sqrt(np.longdouble(128))+lp[None]+np.log(np.longdouble(64));w=np.exp(logits-scale[:,None,None]);ZH=w.sum(-1);NH=np.einsum('qpg,pgd->qpd',w,nu);N=d['N'].cpu().numpy().astype(np.longdouble);Z=d['Z'].cpu().numpy().astype(np.longdouble);OD=d['dense'].cpu().numpy().astype(np.longdouble)
    def obj(mask):
     out=np.where(mask[None,:,None],N,NH).sum(1)/np.where(mask[None],Z,ZH).sum(1)[:,None];return ((OD-out)**2).sum()/((OD**2).sum()+np.longdouble(1e-12))
    mask=v['mask'].numpy()
    if name!='Baseline':
     qi=np.linspace(0,63,16).round().astype(int);qq=q[qi];st=s.stats[0].double().cpu().numpy().copy();oldkeys=s.keys[0].double().cpu().numpy();changed=np.any(keys!=oldkeys,axis=(1,2));km=d['k'][0].double().cpu().numpy().reshape(-1,4,16,128).mean(2);nb=np.sqrt(((km-keys)**2).sum(-1));st[changed,:,8]=torch.from_numpy(nb.astype(np.float64)).bfloat16().double().numpy()[changed]
     zz=logits[qi];ww=np.exp(zz-zz.max((1,2),keepdims=True));prob=ww/ww.sum((1,2),keepdims=True);alpha=prob.sum(-1);q2=(qq*qq).reshape(16,8,16).sum(-1)/128;qn=np.sqrt((qq*qq).sum(-1))/np.sqrt(np.longdouble(128));var=np.einsum('qj,pgj->qpg',q2,st[:,:,:8]);bb=qn[:,None,None]*st[None,:,:,8];bp=st[:,1,11];ccv=bp+bp.max();x0=alpha*st[None,:,0,11]+(abs(prob-alpha[...,None]/4)*st[None,:,:,9]).sum(-1);x1=(prob*np.sqrt(var)*st[None,:,:,10]).sum(-1);x2=ccv[None]*(prob*bb).sum(-1);x3=ccv[None]*(prob*(var+bb*bb)).sum(-1);risk=(np.stack([x0,x1,x2,x3],-1)*m.theta[0].double().cpu().numpy()).sum(-1).mean(0);risk[d['own']]=np.inf;expected=np.zeros(len(mask),dtype=bool);expected[np.argsort(risk)[-int(mask.sum()):]]=True;assert np.array_equal(mask,expected),(seed,event,name,'independent Risk mismatch')
    j=obj(mask);jerr=abs(float(j)-row['new_risk']);best=int(v['gains'].argmax());new=mask.copy();new[int(v['remove'][best])]=False;new[int(v['add'][best])]=True;g=j-obj(new);gerr=abs(float(g)-float(v['gains'][best]));assert jerr<1e-7+1e-7*row['new_risk'] and gerr<1e-7+1e-7*abs(float(g))
    checks.append(dict(seed=seed,event=event,method=name,nmse_error=jerr,oracle_action_gain_error=gerr,positive_denominator=bool((ZH.sum(1)>0).all())))
  done=json.loads((dest/'complete.json').read_text());assert done['C_sha']==sha(checkpoint('C',seed)) and done['protocol_sha']==sha(DEST/'protocol.json')
 dump(DEST/'fixed_mask_decomposition.json',fixed);dump(DEST/'independent_audit.json',dict(counts=counts,provenance=provenance,baseline_checks=Bchecks,checks=checks,models_unchanged=True,network_training=False,test_accessed=False));print('AUDIT PASSED',len(counts),len(checks),len(provenance),flush=True)
if __name__=='__main__':main()
