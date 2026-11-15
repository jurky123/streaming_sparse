from .core import *
def category(i,b,d,g):
 if g>1e-12:return 'positive_gain'
 if g>=-1e-12:return 'tie'
 if i<=0:return 'nonpositive_interaction'
 if i-b<=0:return 'self_energy_exceeds_interaction'
 return 'denominator_reversal'
def metrics(s,G,J):
 ix=s.argmax(1);g=G[np.arange(len(G)),ix];gate=s.max(1)>0;actual=np.where(gate,g,0);oracle=np.maximum(G.max(1),0);eps=1e-12;truth=G>eps;neg=G< -eps;valid=truth|neg;pred=s>0
 recalls=[]
 if truth.any():recalls.append(float(pred[truth].mean()))
 if neg.any():recalls.append(float((~pred[neg]).mean()))
 rho=[corr(s[i],G[i],True) for i in range(len(G))]
 return dict(states=len(G),candidate_sign_accuracy=float(np.mean((pred==truth)[valid])) if valid.any() else None,balanced_sign_accuracy=float(np.mean(recalls)) if recalls else None,positive_recall=float(pred[truth].mean()) if truth.any() else None,false_positive_rate=float(pred[neg].mean()) if neg.any() else None,rank_spearman=float(np.mean([x for x in rho if x is not None])) if any(x is not None for x in rho) else None,forced_gain=float(g.mean()),forced_relative_pct=float(np.mean(g/J)*100),gated_gain=float(actual.mean()),gated_relative_pct=float(np.mean(actual/J)*100),gated_degradation_rate=float(np.mean(actual< -eps)),gated_relative_p10=float(np.quantile(actual/J,.1)*100),positive_filtered_gain=float(np.maximum(g,0).mean()),positive_filtered_relative_pct=float(np.mean(np.maximum(g,0)/J)*100),oracle_gain=float(oracle.mean()),regret=float(np.mean(oracle-actual)),forced_best_action_regret=float(np.mean(oracle-g)),top1_hit=float(np.mean(ix==G.argmax(1))),executed_fraction=float(gate.mean()))
def run():
 lock=json.loads((OUT/'model_lock.json').read_text());modelhash=sha(OUT/'model_lock.json');models={r['representation']:r for r in lock['models']};performance=[];choices=[];mechanism=[];decomp=[]
 for eid in ids():
  d=np.load(OUT/'events'/eid/'terms.npz');G=d['gains'];J=d['nmse'];I=d['I'];B=d['B'];D=d['D'];S=len(G);ident=dict(event_id=eid,split=split(eid));selectors={'all':np.arange(S),'risk':np.array([0]),'post_risk':np.arange(1,S)};scores={}
  for rep in REPS:
   scores[rep+'__baseline']=score(d,rep,1.);scores[rep+'__fitted']=score(d,rep,models[rep]['lambda_fitted']);scores[rep+'__selected']=score(d,rep,models[rep]['lambda_selected']);scores[rep+'__oracle_current_z']=d[rep+'_current_z'];scores[rep+'__oracle_full_z']=d[rep+'_full_z']
  scores['oracle_true_gain']=G
  for name,s in scores.items():
   for scope,sel in selectors.items():performance.append(ident|dict(method=name,scope=scope,**metrics(s[sel],G[sel],J[sel])))
   ix=s.argmax(1)
   for i,j in enumerate(ix):
    choices.append(ident|dict(method=name,state=str(d['names'][i]),state_index=i,action=int(j),remove=int(d['remove'][j]),add=int(d['add'][j]),pred_score=float(s[i,j]),execute=bool(s[i,j]>0),I=float(I[i,j]),B=float(B[i,j]),D=float(D[i,j]),fixed_denominator_gain=float(I[i,j]-B[i,j]),true_gain=float(G[i,j]),nmse=float(J[i]),error_category=category(I[i,j],B[i,j],D[i,j],G[i,j])))
  for rep in REPS:
   ip=d[rep+'_I'];bp=d[rep+'_B']
   for scope,sel in selectors.items():
    it=I[sel];bt=B[sel];dt=D[sel];g=G[sel];pp=ip[sel];bb=bp[sel];valid=abs(it)>1e-12;pos=(it>1e-12)&(pp>1e-15);pvalid=abs(d['I16'][sel])>1e-12;fixed=it-bt
    mechanism.append(ident|dict(representation=rep,scope=scope,interaction_sign_accuracy=float(np.mean((pp[valid]>0)==(it[valid]>0))) if valid.any() else None,interaction_sign_accuracy_matching16=float(np.mean((pp[pvalid]>0)==(d['I16'][sel][pvalid]>0))) if pvalid.any() else None,interaction_pearson=corr(pp,it),energy_pearson=corr(bb,bt),relative_energy_underweighted_fraction=float(np.mean((bb[pos]/pp[pos])<(bt[pos]/it[pos]))) if pos.any() else None,positive_fixed_gain_fraction=float(np.mean(fixed>1e-12)),denominator_harmful_flip_fraction=float(np.mean((fixed>1e-12)&(g< -1e-12))),denominator_rescue_fraction=float(np.mean((fixed< -1e-12)&(g>1e-12))),denominator_abs_share=float(np.mean(abs(dt)/(abs(it)+bt+abs(dt)+1e-30))),proxy_to_true_energy_ratio_median=float(np.median(d[rep+'_B_oracle_units'][sel]/np.maximum(d['B16'][sel],1e-30)))))
  for scope,sel in selectors.items():
   for label,keep in [('all',np.ones_like(G[sel],bool)),('true_positive',G[sel]>1e-12),('true_negative',G[sel]< -1e-12)]:
    if keep.any():decomp.append(ident|dict(scope=scope,candidates=label,count=int(keep.sum()),I_mean=float(I[sel][keep].mean()),B_mean=float(B[sel][keep].mean()),D_mean=float(D[sel][keep].mean()),gain_mean=float(G[sel][keep].mean())))
 table('performance_per_event.csv',performance);table('selected_actions.csv',choices);table('mechanism_per_event.csv',mechanism);table('decomposition_per_event.csv',decomp)
 def aggregate(rows,keys):
  out=[]
  for vals in sorted({tuple(r[k] for k in keys) for r in rows}):
   rr=[r for r in rows if tuple(r[k] for k in keys)==vals];v=dict(zip(keys,vals));v['events']=len(rr)
   for k in rr[0]:
    if k in keys or k=='event_id':continue
    a=[r[k] for r in rr if r[k] is not None]
    if a and isinstance(a[0],(int,float,np.floating)):v[k]=float(np.mean(a))
   if 'oracle_gain' in v:v['pooled_gated_recovery']=v['gated_gain']/v['oracle_gain'] if v['oracle_gain']>1e-30 else None;v['pooled_positive_filtered_recovery']=v['positive_filtered_gain']/v['oracle_gain'] if v['oracle_gain']>1e-30 else None
   out.append(v)
  return out
 perf=aggregate(performance,['split','scope','method']);mech=aggregate(mechanism,['split','scope','representation']);terms=aggregate(decomp,['split','scope','candidates']);table('performance_summary.csv',perf);table('mechanism_summary.csv',mech);table('decomposition_summary.csv',terms)
 # Selected E15 actions: arithmetic error categories, retain state and event weights.
 categories=[]
 for sp in ['train','validation','evaluation']:
  for scope in ['all','risk','post_risk']:
   for rep in REPS:
    rr=[r for r in choices if r['split']==sp and r['method']==rep+'__baseline' and (scope=='all' or (r['state_index']==0)==(scope=='risk'))];weights=[]
    for r in rr:weights.append(1/sum(x['event_id']==r['event_id'] for x in rr))
    w=np.array(weights);bad=np.array([r['true_gain']< -1e-12 and r['execute'] for r in rr]);total=float(w[bad].sum())
    for cat in ['nonpositive_interaction','self_energy_exceeds_interaction','denominator_reversal']:
     yes=np.array([r['error_category']==cat for r in rr])&bad;categories.append(dict(split=sp,scope=scope,representation=rep,category=cat,event_weighted_bad_states=total,share_of_executed_errors=float(w[yes].sum()/total) if total else None,raw_bad_states=int(bad.sum()),raw_category_states=int(yes.sum())))
 table('E15_error_categories.csv',categories)
 assert sha(OUT/'model_lock.json')==modelhash
 dump(OUT/'summary.json',dict(performance=perf,mechanism=mech,decomposition=terms,error_categories=categories,model_lock_sha256=modelhash));print(json.dumps([r for r in perf if r['split']=='evaluation' and r['scope']=='risk'],indent=2))
if __name__=='__main__':run()
