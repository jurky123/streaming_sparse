"""Frozen beta reuse: context weighting, one mechanistic feature, loss alignment."""
from ..e30_beta_diagnostics.run import *
from ..e25_decision_diagnostics.report import aggregate
E30=DEST
DEST=ROOT/'artifacts/e31_beta_reuse_v1'
SCHEMES=['A0','A1','A2'];CATEGORIES=['tile','step','forward']
PRIMARY=['Predicted','EventOracle','True']+[s+'-'+m for s in SCHEMES for m in ['Shared','LOTO']]

def fit_context(fs,ls,ys,scale,balanced):
 if balanced:
  ws=[(l-.5*torch.logsumexp(2*l,0)[None]).exp() for l in ls]
 else:
  mx=torch.cat(ls).max(0).values;ws=[(l-mx[None]).exp() for l in ls]
 X=torch.cat([w[...,None]*f for w,f in zip(ws,fs)]).permute(1,0,2)
 Y=torch.cat([w*y for w,y in zip(ws,ys)]).T[...,None]
 co,_,audit=ridge_fit(X,Y,scale)
 return co,audit

def donors(records,sig,j,kind):
 e=records[sig[j]]['ev'];out=[]
 for k,i in enumerate(sig):
  if k==j:continue
  d=records[i]['ev']
  if kind=='tile' and d['chunk']==e['chunk'] and d['step']==e['step'] and d['tile']!=e['tile']:out.append(k)
  if kind=='step' and d['chunk']==e['chunk'] and d['step']!=e['step']:out.append(k)
  if kind=='forward' and d['chunk']<e['chunk']:out.append(k)
 return out

def loss_stats(beta,bt,lm,select):
 e=beta[:,select].double()-bt[:,select];lw=2*lm[:,select];w=lw.exp();wn=lw.flatten().softmax(0).reshape_as(lw)
 return dict(beta_mae=float(e.abs().mean()),beta_rmse=float(e.square().mean().sqrt()),quality_sum=float((w*e.square()).sum()),quality_weight=float(w.sum()),quality_weight_normalized=float((wn*e.square()).sum()))

def decision_loss(pred,teacher,gscale):
 x=torch.as_tensor(pred,dtype=torch.float64);y=torch.as_tensor(teacher,dtype=torch.float64)
 scale=torch.maximum(y.square().mean(1).sqrt(),torch.full((len(y),),float(gscale),dtype=y.dtype))
 xx=x/scale[:,None];yy=y/scale[:,None];h=nn.functional.smooth_l1_loss(xx,yy)
 xx=torch.cat([xx,torch.zeros(len(xx),1)],1);yy=torch.cat([yy,torch.zeros(len(yy),1)],1)
 tlog=torch.log_softmax(yy/.3,1);slog=torch.log_softmax(xx/.3,1);kl=(tlog.exp()*(tlog-slog)).sum(1).mean()
 return dict(gain_huber=float(h),rank_noop_kl=float(kl),decision_loss=float(h+.1*kl),teacher_scale_min=float(scale.min()),teacher_scale_max=float(scale.max()))

@torch.no_grad()
def run(seed,limit_groups=None):
 torch.set_num_threads(2);dest=DEST/f's{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'complete.json').exists()
 assert (E30/f's{seed}'/'complete.json').exists()
 bundle=bundle_for('C',seed);fit=torch.load(OUT/'fit.pt',weights_only=True);scale=fit['bscale'];gscale=fit['gscale'];qs=torch.linspace(0,63,16,device='cuda').round().long();paths=files('development');start=time.time()
 groups=collections.defaultdict(list);identpath=ROOT/'artifacts/e28_local_response_v1'/f's{seed}'/'identities.json';ident=json.loads(identpath.read_text());ig=collections.defaultdict(list)
 for x in ident:ig[tuple(x['group'])].append(x)
 for i,path in enumerate(paths):
  ev=torch.load(path,map_location='cpu',weights_only=True);groups[(ev['trajectory'],ev['layer'],ev['head'])].append((i,path))
 rows=[];quality=[];conditions=[];losses=[];coverage=[];fitaudit=[];weightrows=[];audit=[];teachers=[];count=0
 for gi,(group,items0) in enumerate(groups.items()):
  if limit_groups is not None and gi>=limit_groups:break
  records=[];emap={}
  for idx,path in items0:
   x=torch.load(E30/f's{seed}'/f'beta_{idx:04d}.pt',map_location='cuda',weights_only=True);p=np.load(E30/f's{seed}'/f'panel_{idx:04d}.npz');d,s,c,_,_,_,_=prepare_local(bundle,path)
   assert x['event']==str(path.relative_to(OUT));F=x['F'];q2=d['q'].double().square().reshape(64,8,16).sum(-1)/128
   v=(torch.einsum('qj,pgj->qpg',q2,s.stats[0,:,:,:8].double())*c['logit'].softmax(-1)).sum(-1);Fv=F.clone();Fv[...,4]=v
   mask=torch.as_tensor(p['states'][0],device='cuda');r=torch.as_tensor(p['initial_remove'],device='cuda');a=torch.as_tensor(p['initial_add'],device='cuda');candidate=torch.zeros_like(mask);candidate[r]=True;candidate[a]=True
   norm=d['dense'][qs].square().sum()/16+1e-12;bt=x['beta_true'];lm=x['logmass'];base=x['betas']['Predicted'];event=x['betas']['OracleCoeff64'];hist=torch.arange(len(mask),device='cuda')<d['hp'];hybrid=torch.where(hist[None],base,event)
   betas={'Predicted':base,'EventOracle':event,'True':bt,'HistoryPred-CurrentOracle':hybrid}
   supports={};donorcounts={}
   for scheme in SCHEMES:
    for mode in ['Shared','LOTO']:betas[scheme+'-'+mode]=event.clone()
    for kind in CATEGORIES:
     betas[scheme+'-'+kind]=hybrid.clone();betas[scheme+'-matched-'+kind]=hybrid.clone()
   for kind in CATEGORIES:supports[kind]=torch.zeros_like(mask);donorcounts[kind]=torch.zeros_like(mask,dtype=torch.long)
   supports['matched']=torch.zeros_like(mask)
   rec=dict(idx=idx,event=x['event'],ev=d['ev'],path=path,F=F,Fv=Fv,v=v,lm=lm,bt=bt,rp=x['predicted_response'],center=x['center'],hp=d['hp'],hist=hist,mask=mask,r=r,a=a,candidate=candidate,norm=norm,J=float(objective(c,d['N'],d['Z'],d['dense'],mask)),g=torch.as_tensor(p['true_initial_gain'],device='cuda'),panel={k:p[k] for k in ['states','remove','add','true_gain']},betas=betas,supports=supports,donorcounts=donorcounts)
   emap[x['event']]=len(records);records.append(rec)
   # Inputs and existing E30 identities unchanged. Independent feature reference follows in audit.
   assert torch.allclose(lm,c['logzh']-c['A'].log()[:,None],atol=1e-9,rtol=1e-9)
   audit.append(dict(seed=seed,event=x['event'],feature_min=float(v.min()),feature_max=float(v.max()),feature_finite=bool(torch.isfinite(v).all()),summary_sha_checked=True))
   rec['summary_sha']=[digest(*(getattr(s,n)[0,p] for n in ['keys','mean','detail','stats'])) for p in range(d['hp'])]
  signatures=collections.defaultdict(list)
  for x in ig[group]:
   occ=[(emap[o['event']],o['page']) for o in x['occurrences']]
   for i,p in occ:assert records[i]['summary_sha'][p]==x['summary_sha']
   signatures[tuple(i for i,p in occ)].append(occ)
  for sig,items in signatures.items():
   assert len(sig)>1
   ix=[torch.tensor([o[j][1] for o in items],device='cuda') for j in range(len(sig))];ls=[records[i]['lm'][:,pp] for i,pp in zip(sig,ix)];ys=[records[i]['bt'][:,pp] for i,pp in zip(sig,ix)]
   logw=torch.stack([torch.logsumexp(2*l,0) for l in ls]);share=logw.softmax(0);dom=share.max(0).values;eff=1/share.square().sum(0)
   weightrows.append(dict(seed=seed,group=str(group),pages=len(items),occurrences=len(sig),max_event_weight_mean=float(dom.mean()),max_event_weight_median=float(dom.median()),effective_contexts_mean=float(eff.mean()),dominated_gt90_pages=int((dom>.9).sum()),dominated_gt99_pages=int((dom>.99).sum())))
   for scheme in SCHEMES:
    fs=[records[i]['Fv' if scheme=='A2' else 'F'][:,pp] for i,pp in zip(sig,ix)];balanced=scheme!='A0';co,fa=fit_context(fs,ls,ys,scale,balanced);fitaudit.append(dict(seed=seed,scheme=scheme,mode='Shared',pages=len(items),contexts=len(sig),**fa))
    for j,(i,pp) in enumerate(zip(sig,ix)):
     rec=records[i];rec['betas'][scheme+'-Shared'][:,pp]=torch.einsum('qpf,pfd->qpd',fs[j],co)[...,0]
     other=[k for k in range(len(sig)) if k!=j];cl,fa=fit_context([fs[k] for k in other],[ls[k] for k in other],[ys[k] for k in other],scale,balanced);rec['betas'][scheme+'-LOTO'][:,pp]=torch.einsum('qpf,pfd->qpd',fs[j],cl)[...,0];fitaudit.append(dict(seed=seed,scheme=scheme,mode='LOTO',event=rec['event'],pages=len(pp),contexts=len(other),**fa))
     dd={kind:donors(records,sig,j,kind) for kind in CATEGORIES};matched=all(dd.values())
     if matched:rec['supports']['matched'][pp]=True
     for kind,kk in dd.items():
      if not kk:continue
      rec['supports'][kind][pp]=True;rec['donorcounts'][kind][pp]=len(kk)
      cc,fa=fit_context([fs[k] for k in kk],[ls[k] for k in kk],[ys[k] for k in kk],scale,balanced);val=torch.einsum('qpf,pfd->qpd',fs[j],cc)[...,0];rec['betas'][scheme+'-'+kind][:,pp]=val
      if matched:rec['betas'][scheme+'-matched-'+kind][:,pp]=val
  for rec in records:
   idx=rec['idx'];p=rec['panel'];r=rec['r'];a=rec['a'];mask=rec['mask'];g=rec['g'];meta=dict(seed=seed,event=rec['event'],**{k:rec['ev'][k] for k in ['trajectory','layer','chunk','step','head','tile']});out={};initial={};allbeta={};candid=rec['candidate'];hist=rec['hist']
   # All loss comparisons use the same teacher response and the same saved true beta.
   teacher=np.stack([local_scores(rec['rp'][qs],rec['bt'][qs],rec['lm'][qs],rec['center'][qs],torch.as_tensor(st,device='cuda'),torch.as_tensor(p['remove'],device='cuda'),torch.as_tensor(p['add'],device='cuda'),rec['norm'])[0].cpu().numpy() for st in p['states']])
   for st,(tg,t) in enumerate(zip(teacher,p['true_gain'])):
    j=int(tg.argmax());teachers.append(dict(**meta,state=st,teacher_execute=bool(tg[j]>0),teacher_true_gain=float(t[j]) if tg[j]>0 else 0.,teacher_proposal_gain=float(t[j]),actual_oracle=max(0.,float(t.max())),teacher_regret=max(0.,float(t.max()))-(float(t[j]) if tg[j]>0 else 0.)))
   for method,bb in rec['betas'].items():
    sc,ind=local_scores(rec['rp'][qs],bb[qs],rec['lm'][qs],rec['center'][qs],mask,r,a,rec['norm']);initial[method]=sc.cpu().numpy();allbeta[method]=bb[qs].cpu()
    category='full';support=int(hist.sum())
    for kind in CATEGORIES:
     if method.endswith('-'+kind):category='matched' if '-matched-' in method else kind;support=int(rec['supports'][category].sum())
    for policy,score in [('Set',sc),('Independent',ind)]:
     j=int(score.argmax());execute=bool(score[j]>0);m=mask.clone()
     if execute:m[r[j]]=False;m[a[j]]=True
     assert int(m.sum())==int(mask.sum()) and bool(m[rec['ev']['own']])
     rows.append(dict(**meta,method=method,policy=policy,initial=rec['J'],oracle_gain=max(0.,float(g.max())),proposal_gain=float(g[j]),hindsight_gain=max(0.,float(g[j])),executed_gain=float(g[j]) if execute else 0.,execute=execute,remove=int(r[j]),add=int(a[j]),predicted_gain=float(score[j]),coverage_kind=category,covered_history_pages=support,history_pages=rec['hp']))
    if method not in PRIMARY:continue
    ps=np.stack([local_scores(rec['rp'][qs],bb[qs],rec['lm'][qs],rec['center'][qs],torch.as_tensor(st,device='cuda'),torch.as_tensor(p['remove'],device='cuda'),torch.as_tensor(p['add'],device='cuda'),rec['norm'])[0].cpu().numpy() for st in p['states']]);out[method]=ps;conditions.append(dict(**meta,method=method,**conditional_metrics(p['true_gain'],ps)))
    for qname,qq in [('M16',qs),('Q64',torch.arange(64,device='cuda'))]:
     for region,sel in [('all',torch.ones_like(mask)),('candidate',candid),('background',~candid),('history',hist)]:quality.append(dict(**meta,method=method,queries=qname,region=region,**loss_stats(bb[qq],rec['bt'][qq],rec['lm'][qq],sel)))
    dec=decision_loss(ps,teacher,gscale);j=int(sc.argmax());tp=torch.as_tensor(initial['True'],device='cuda') if 'True' in initial else local_scores(rec['rp'][qs],rec['bt'][qs],rec['lm'][qs],rec['center'][qs],mask,r,a,rec['norm'])[0]
    losses.append(dict(**meta,method=method,**dec,gain_spearman=float(spearmanr(sc.cpu().numpy(),g.cpu().numpy()).statistic),teacher_gain_spearman=float(spearmanr(sc.cpu().numpy(),tp.cpu().numpy()).statistic),sign_accuracy=float(((sc>0)==(g>1e-12)).double().mean()),positive_recall=float(((sc>0)&(g>1e-12)).sum()/(g>1e-12).sum().clamp_min(1)),true_regret=float(g.max().clamp_min(0)-(g[j] if sc[j]>0 else 0)),proposal_regret=float(g.max().clamp_min(0)-g[j]),teacher_score_mse=float(np.mean((ps-teacher)**2))))
   for kind,sel in rec['supports'].items():
    mass=rec['lm'][qs].exp();coverage.append(dict(**meta,kind=kind,pages=int(sel.sum()),history_pages=rec['hp'],history_mass_fraction=float(mass[:,sel].sum()/mass[:,hist].sum().clamp_min(1e-300)),candidate_pages=int((sel&candid).sum()),candidate_history_pages=int((hist&candid).sum())))
   old=torch.load(E30/f's{seed}'/f'beta_{idx:04d}.pt',weights_only=True)
   for mode in ['Shared','LOTO']:
    diff=float((rec['betas']['A0-'+mode].cpu()-old['betas'][mode]).abs().max());assert diff<1e-7,(seed,idx,mode,diff)
   np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=rec['event'],states=p['states'],remove=p['remove'],add=p['add'],true_gain=p['true_gain'],true_initial_gain=g.cpu().numpy(),initial_remove=r.cpu().numpy(),initial_add=a.cpu().numpy(),**out,**{'initial_'+k:v for k,v in initial.items()})
   torch.save(dict(event=rec['event'],variance_feature=rec['v'].cpu(),betas_m16=allbeta,supports={k:v.cpu() for k,v in rec['supports'].items()},donorcounts={k:v.cpu() for k,v in rec['donorcounts'].items()}),dest/f'fits_{idx:04d}.pt');count+=1
  for name,data in [('decisions',rows),('quality',quality),('conditional',conditions),('losses',losses),('coverage',coverage),('fit_audit',fitaudit),('weight_dominance',weightrows),('audit',audit),('teacher_gap',teachers)]:
   for row in data:
    for key,v in row.items():
     if isinstance(v,float) and not math.isfinite(v):row[key]=None
   dump(dest/(name+'.json'),data)
  dump(dest/'status.json',dict(seed=seed,events=count,groups=gi+1,seconds=time.time()-start));print('GROUP',seed,gi+1,count,round(time.time()-start),flush=True)
 if limit_groups is None:
  assert count==288;dump(dest/'complete.json',dict(seed=seed,events=count,seconds=time.time()-start,training=False,test_accessed=False,protocol_sha=sha(DEST/'protocol.json'),identity_sha=sha(identpath),reader_sha=sha(E29/f's{seed}'/'selected.pt'),C_sha=sha(checkpoint('C',seed)),phi_sha=sha(E27/f's{seed}'/'selected.pt')))
 print('DONE',seed,count,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--limit-groups',type=int);a=p.parse_args();run(a.seed,a.limit_groups)
