"""Independent controls: context support, NumPy fits, positive-mixture gains."""
from .run import *

@torch.no_grad()
def main():
 torch.set_num_threads(2);checks=[];fitchecks=[];coveragechecks=[];counts=[];paths=files('development');qs=np.linspace(0,63,16).round().astype(int)
 for seed in [17,29]:
  dest=DEST/f's{seed}';rows=json.loads((dest/'decisions.json').read_text());panels={};events={};idxmap={}
  for i in range(288):
   p=np.load(dest/f'panel_{i:04d}.npz');event=str(p['event']);panels[event]=p;idxmap[event]=i
  for r in rows:
   events[r['event']]=r;p=panels[r['event']];m=p['states'][0].copy();own=len(m)-72+r['tile'];assert m[r['remove']] and not m[r['add']] and r['remove']!=own;assert np.any((p['initial_remove']==r['remove'])&(p['initial_add']==r['add']))
   if r['execute']:m[r['remove']]=False;m[r['add']]=True
   assert m.sum()==max(1,int(len(m)*.1)) and m[own]
  counts.append(dict(seed=seed,decisions=len(rows),count_own_pool_valid=True))
  identities=json.loads((ROOT/'artifacts/e28_local_response_v1'/f's{seed}'/'identities.json').read_text());expected={e:{k:np.zeros(len(panels[e]['states'][0]),dtype=int) for k in CATEGORIES} for e in events}
  for item in identities:
   for o in item['occurrences']:
    e=events[o['event']];pp=o['page'];others=[x for x in item['occurrences'] if x['event']!=o['event']]
    for x in others:
     v=events[x['event']]
     if v['chunk']==e['chunk'] and v['step']==e['step'] and v['tile']!=e['tile']:expected[o['event']]['tile'][pp]+=1
     if v['chunk']==e['chunk'] and v['step']!=e['step']:expected[o['event']]['step'][pp]+=1
     if v['chunk']<e['chunk']:expected[o['event']]['forward'][pp]+=1
  for e,i in idxmap.items():
   x=torch.load(dest/f'fits_{i:04d}.pt',weights_only=True);intersection=np.ones(len(panels[e]['states'][0]),dtype=bool)
   for kind in CATEGORIES:
    assert np.array_equal(expected[e][kind],x['donorcounts'][kind].numpy());assert np.array_equal(expected[e][kind]>0,x['supports'][kind].numpy());intersection&=expected[e][kind]>0
   assert np.array_equal(intersection,x['supports']['matched'].numpy())
  coveragechecks.append(dict(seed=seed,events=288,all_supports_donors_exact=True,forward_strictly_earlier=True))
  # Independent ridge on representative history pages, including cross-chunk.
  chosen=identities[:3]+[x for x in identities if len({events[o['event']]['chunk'] for o in x['occurrences']})>1][:3]
  cache={};newcache={}
  def get(e):
   i=idxmap[e]
   if e not in cache:cache[e]=torch.load(E30/f's{seed}'/f'beta_{i:04d}.pt',weights_only=True);newcache[e]=torch.load(dest/f'fits_{i:04d}.pt',weights_only=True)
   return cache[e],newcache[e]
  for item in chosen:
   hold=item['occurrences'][-1];he=hold['event'];hp=hold['page'];hx,hn=get(he)
   for scheme in SCHEMES:
    for mode in ['Shared','LOTO']+CATEGORIES:
     candidates=[]
     for o in item['occurrences']:
      e=o['event'];v=events[e];h=events[he]
      include=(mode=='Shared' or mode=='LOTO' and e!=he or mode=='tile' and e!=he and v['chunk']==h['chunk'] and v['step']==h['step'] and v['tile']!=h['tile'] or mode=='step' and v['chunk']==h['chunk'] and v['step']!=h['step'] or mode=='forward' and v['chunk']<h['chunk'])
      if include:candidates.append(o)
     if not candidates:continue
     fs=[];ls=[];ys=[]
     for o in candidates:
      x,n=get(o['event']);p=o['page'];f=x['F'][:,p].numpy().copy()
      if scheme=='A2':f[:,4]=n['variance_feature'][:,p].numpy()
      fs.append(f);ls.append(x['logmass'][:,p].numpy());ys.append(x['beta_true'][:,p].numpy())
     if scheme=='A0':mx=max(l.max() for l in ls);ws=[np.exp(l-mx) for l in ls]
     else:ws=[np.exp(l-l.max())/np.sqrt(np.square(np.exp(l-l.max())).sum()) for l in ls]
     X=np.concatenate([f*w[:,None] for f,w in zip(fs,ws)]);Y=np.concatenate([y*w for y,w in zip(ys,ws)]);G=X.T@X;co=np.linalg.solve(G+1e-6*np.trace(G)/5*np.eye(5),X.T@Y);hf=hx['F'][qs,hp].numpy().copy()
     if scheme=='A2':hf[:,4]=hn['variance_feature'][qs,hp].numpy()
     pred=hf@co;saved=hn['betas_m16'][scheme+'-'+mode][:,hp].numpy();err=float(np.max(abs(pred-saved)));assert err<1e-6,(seed,scheme,mode,err);fitchecks.append(dict(seed=seed,event=he,page=hp,scheme=scheme,mode=mode,n_contexts=len(candidates),error=err))
  bundle=bundle_for('C',seed)
  worst=min([r for r in rows if r['method']=='A2-LOTO' and r['policy']=='Set' and r['initial']>1e-12],key=lambda r:r['executed_gain']/r['initial'])['event']
  for idx in sorted(set([0,47,143,144,230,idxmap[worst]])):
   path=paths[idx];d,s,c,_,_,_,_=prepare_local(bundle,path);x=torch.load(E30/f's{seed}'/f'beta_{idx:04d}.pt',weights_only=True);n=torch.load(dest/f'fits_{idx:04d}.pt',weights_only=True);event=x['event'];p=panels[event];m=p['states'][0];norm=float(d['dense'][qs].square().sum()/16+1e-12)
   # Key feature recomputed by explicit blocks/subgroups, independent of einsum implementation.
   q=d['q'].cpu().numpy().astype(np.float64);st=s.stats[0,:,:,:8].double().cpu().numpy();log=c['logit'].cpu().numpy();pi=np.exp(log-log.max(-1,keepdims=True));pi/=pi.sum(-1,keepdims=True);v=np.zeros((64,len(m)))
   for g in range(4):
    for j in range(8):v+=pi[:,:,g]*np.square(q[:,j*16:(j+1)*16]).sum(-1)[:,None]/128*st[None,:,g,j]
   verr=float(np.max(abs(v-n['variance_feature'].numpy())));assert verr<1e-8
   N=d['N'].cpu().numpy().astype(np.longdouble);Z=d['Z'].cpu().numpy().astype(np.longdouble);NH=c['nh'].cpu().numpy().astype(np.longdouble);ZH=c['zh'].cpu().numpy().astype(np.longdouble);OD=d['dense'].cpu().numpy().astype(np.longdouble)
   def obj(mask):
    O=np.where(mask[None,:,None],N,NH).sum(1)/np.where(mask[None],Z,ZH).sum(1)[:,None];return ((O-OD)**2).sum()/((OD**2).sum()+np.longdouble(1e-12))
   J=obj(m);l=x['logmass'][qs].numpy().astype(np.longdouble);cen=x['center'][qs].numpy().astype(np.longdouble);vpage=cen+x['predicted_response'][qs].numpy().astype(np.longdouble)
   def sm(z):
    w=np.exp(z-z.max(1,keepdims=True));return w/w.sum(1,keepdims=True)
   for row in [r for r in rows if r['event']==event and r['policy']=='Set']:
    b=n['betas_m16'][row['method']].numpy().astype(np.longdouble);target=(sm(l+b)[...,None]*vpage).sum(1)
    def h(mask):return (sm(np.where(mask[None],l+b,l))[...,None]*np.where(mask[None,:,None],vpage,cen)).sum(1)
    new=m.copy();new[row['remove']]=False;new[row['add']]=True;H=h(m);delta=h(new)-H;gpred=(2*(target-H)*delta-delta*delta).sum(-1).mean()/norm;gt=J-obj(new);pe=abs(float(gpred)-row['predicted_gain']);te=abs(float(gt)-row['proposal_gain']);assert pe<1e-8+1e-6*abs(float(gpred)) and te<1e-8,(seed,idx,row['method'],pe,te)
    checks.append(dict(seed=seed,event=event,method=row['method'],predicted_gain_error=pe,true_gain_error=te,feature_error=verr))
  done=json.loads((dest/'complete.json').read_text());assert done['reader_sha']==sha(E29/f's{seed}'/'selected.pt') and done['C_sha']==sha(checkpoint('C',seed)) and done['phi_sha']==sha(E27/f's{seed}'/'selected.pt')
 losses=json.loads((DEST/'s17'/'losses.json').read_text())+json.loads((DEST/'s29'/'losses.json').read_text());zero=max(abs(r['decision_loss']) for r in losses if r['method']=='True');assert zero<1e-12
 dump(DEST/'independent_audit.json',dict(counts=counts,checks=checks,ridge=fitchecks,coverage=coveragechecks,all_frozen_hashes_unchanged=True,true_teacher_decision_loss_max=zero,test_accessed=False,training=False));print('AUDIT PASSED',len(checks),len(fitchecks),flush=True)
if __name__=='__main__':main()
