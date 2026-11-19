from .core import *
import tarfile,shutil,hashlib

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,float) and not math.isfinite(x):return None
 return x

def agg(rr):
 z=aggregate(rr);z.update(H_loss=sum(max(0.,-r['executed_gain']) for r in rr),H_gain=sum(max(0.,r['executed_gain']) for r in rr),worst_absolute_increase=max(0.,max(-r['executed_gain'] for r in rr)))
 if 'top8' in rr[0]:
  for k in [1,4,8]:z[f'R_top{k}']=sum(r[f'top{k}'] for r in rr)/z['oracle_gain_sum']
  fixed=[r for r in rr if r.get('fixed_margin') is not None]
  if fixed:z.update(fixed_margin_abs_error=float(np.mean([abs(r['fixed_margin']-r['true_fixed_margin']) for r in fixed])),fixed_preference_correct=sum(r['fixed_margin']<=1e-12 for r in fixed),old_harmful=sum(r['old_harmful'] for r in fixed),old_rejected=sum(r['old_harmful'] and r['old_action_rejected'] for r in fixed),old_rejected_new_harmful=sum(r['old_harmful'] and r['old_action_rejected'] and r['executed_gain']< -1e-12 for r in fixed))
 return z

def report():
 decomposition=[];ds=[];summary=[];allrows=[];condition=[];curves=[];info=[];fixedpairs=[]
 for seed in [17,29]:
  rr=json.loads((DEST/f'decomposition_s{seed}.json').read_text());decomposition+=rr
  for dim in ['ALL','layer','trajectory']:
   for val in (['ALL'] if dim=='ALL' else sorted({x[dim] for x in rr})):
    for subset in ['all','different','original_harmful']:
     zz=[x for x in rr if (dim=='ALL' or x[dim]==val) and (subset=='all' or (not x['same_action'] if subset=='different' else x['original_harmful']))]
     row=dict(seed=seed,dimension=dim,value=val,subset=subset,n=len(zz),true16_correct=sum(x['true16_correct'] for x in zz),true64_correct=sum(x['true64_B16_correct'] for x in zz))
     for k in ['prediction_error','query_difference','projection_difference']:
      v=np.array([x[k] for x in zz]);row.update({k+'_sum':float(v.sum()),k+'_abs_sum':float(abs(v).sum()),k+'_median':float(np.median(v)) if len(v) else None,k+'_positive':int((v>1e-12).sum()),k+'_negative':int((v< -1e-12).sum())})
     ds.append(row)
  baseline=json.loads((E38/f'A_s{seed}/decisions.json').read_text());baseline=[dict(r,group='E29',epoch=-1) for r in baseline if r['method']=='E29']
  lookup={x['event']:i for i,x in enumerate(rr)}
  for row in baseline:
   pp=np.load(E30/f"s{seed}/panel_{lookup[row['event']]:04d}.npz");order=np.argsort(-pp['initial_Predicted'],kind='stable');gg=pp['true_initial_gain']
   oldpair=rr[lookup[row['event']]];row['true_fixed_margin']=oldpair['M_true64_full'];row['old_harmful']=oldpair['original_harmful'];row['old_action_rejected']=oldpair['j_pred']<0
   for k in [1,4,8]:row[f'top{k}']=max(0.,float(gg[order[:k]].max()))
  allrows+=baseline
  bdt=[];bdp=[];bn=bd=0;bc=[]
  for idx in range(288):
   pp=np.load(E38/f'A_s{seed}/panel_{idx:04d}.npz');tt=pp['true_gain'];vv=pp['E29'];tc=tt-tt.mean(0);pc=vv-vv.mean(0);bdt.extend((tc-tc.mean(1,keepdims=True)).ravel());bdp.extend((pc-pc.mean(1,keepdims=True)).ravel());bc.append(conditional_metrics(tt,vv)['conditional_corr'])
   for i in range(len(tt)):
    for j in range(i+1,len(tt)):
     yes=(tt[i]*tt[j]<0)&(abs(tt[i])>1e-12)&(abs(tt[j])>1e-12);bd+=int(yes.sum());bn+=int((yes&(vv[i]*vv[j]<0)).sum())
  condition.append(dict(seed=seed,group='E29',epoch=-1,double_centered_corr=stablecorr(np.array(bdt),np.array(bdp)),median_event_corr=float(np.nanmedian([x if x is not None else np.nan for x in bc])),sign_flip_n=bd,sign_flip_recalled=bn,sign_flip_recall=bn/bd if bd else None))
  for group in ['T0','T1']:
   dest=DEST/f'{group}_s{seed}';assert (dest/'training_complete.json').exists()
   curves += [dict(r,seed=seed,group=group) for r in json.loads((dest/'curves.json').read_text())]
   for epoch in [0,1]:
    rows=json.loads((dest/f'development_epoch{epoch}.json').read_text());allrows+=rows
    cr=json.loads((dest/f'development_conditional_epoch{epoch}.json').read_text());info +=[dict(r,seed=seed,group=group,epoch=epoch,split='development') for r in json.loads((dest/f'development_information_epoch{epoch}.json').read_text())]
    dt=[];dp=[];flipn=flipd=0
    for idx,crow in enumerate(cr):
     p=np.load(dest/f'epoch{epoch}_panel_{idx:04d}.npz');t=p['conditional_true'];v=p['conditional'];tt=t-t.mean(0);vv=v-v.mean(0);dt.extend((tt-tt.mean(1,keepdims=True)).ravel());dp.extend((vv-vv.mean(1,keepdims=True)).ravel())
     oldpanel=np.load(E30/f's{seed}/panel_{idx:04d}.npz');d=rr[idx];pair=[d['j_pred'],d['j_oracle']];ids=[]
     for j in pair:
      if j<0:ids.append(-1)
      else:
       match=np.flatnonzero((oldpanel['remove']==oldpanel['initial_remove'][j])&(oldpanel['add']==oldpanel['initial_add'][j]));ids.append(int(match[0]) if len(match) else None)
     if None not in ids:
      tm=(t[:,ids[0]] if ids[0]>=0 else np.zeros(len(t)))-(t[:,ids[1]] if ids[1]>=0 else np.zeros(len(t)));pm=(v[:,ids[0]] if ids[0]>=0 else np.zeros(len(v)))-(v[:,ids[1]] if ids[1]>=0 else np.zeros(len(v)))
      fixedpairs.append(dict(seed=seed,group=group,epoch=epoch,event=d['event'],states=len(t),mae=float(np.mean(abs(pm-tm))),sign_accuracy=float(np.mean((pm>1e-12)==(tm>1e-12))),conditional_corr=stablecorr(tm-tm.mean(),pm-pm.mean())))
     # direct sign transitions, each unordered pair of states
     for i in range(len(t)):
      for j in range(i+1,len(t)):
       yes=(t[i]*t[j]<0)&(abs(t[i])>1e-12)&(abs(t[j])>1e-12);flipd+=int(yes.sum());flipn+=int((yes&(v[i]*v[j]<0)).sum())
    condition.append(dict(seed=seed,group=group,epoch=epoch,double_centered_corr=stablecorr(np.array(dt),np.array(dp)),median_event_corr=float(np.nanmedian([r['conditional_corr'] if r['conditional_corr'] is not None else np.nan for r in cr])),sign_flip_n=flipd,sign_flip_recalled=flipn,sign_flip_recall=flipn/flipd if flipd else None))
   for epoch in [-1,0,1]:
    info +=[dict(r,seed=seed,group=group,epoch=epoch,split='train_monitor') for r in json.loads((dest/f'train_information_epoch{epoch}.json').read_text())]
    for policy in ['Set','Independent']:
     r=[x for x in json.loads((dest/f'train_epoch{epoch}.json').read_text()) if x['policy']==policy];summary.append(dict(seed=seed,group=group,epoch=epoch,split='train_monitor',dimension='ALL',value='ALL',policy=policy,**agg(r)))
 for seed in [17,29]:
  for group,epoch in [('E29',-1),('T0',0),('T0',1),('T1',0),('T1',1)]:
   for policy in ['Set','Independent']:
    rr=[x for x in allrows if x['seed']==seed and x['group']==group and x['epoch']==epoch and x['policy']==policy]
    for dim in ['ALL','trajectory','layer','chunk','step','head']:
     for val in (['ALL'] if dim=='ALL' else sorted({r[dim] for r in rr})):
      ss=[x for x in rr if dim=='ALL' or x[dim]==val];summary.append(dict(seed=seed,group=group,epoch=epoch,split='development',dimension=dim,value=val,policy=policy,**agg(ss)))
 gates=[]
 for seed in [17,29]:
  def get(group,policy='Set'):
   return next(x for x in summary if x['seed']==seed and x['group']==group and x['epoch']==(-1 if group=='E29' else 1) and x['split']=='development' and x['dimension']=='ALL' and x['policy']==policy)
  t0,t1,base,ind=get('T0'),get('T1'),get('E29'),get('T1','Independent');c0=next(x for x in condition if x['seed']==seed and x['group']=='T0' and x['epoch']==1);c1=next(x for x in condition if x['seed']==seed and x['group']=='T1' and x['epoch']==1)
  gain=t1['executed_gain_sum']>t0['executed_gain_sum'];harm=t1['H_loss']<t0['H_loss'];explain=t1['fixed_margin_abs_error']<t0['fixed_margin_abs_error'] or t1['R_top8']>t0['R_top8'] or (c1['double_centered_corr'] or -1)>(c0['double_centered_corr'] or -1)
  mech=gain and harm and explain;traj=all(x['executed_gain_sum']>0 for x in summary if x['seed']==seed and x['group']=='T1' and x['epoch']==1 and x['split']=='development' and x['dimension']=='trajectory' and x['policy']=='Set');usable=mech and t1['R_executed']>=.2 and traj and t1['executed_gain_sum']>ind['executed_gain_sum'] and t1['mean_improvement_pct']>=0 and t1['degraded']<base['degraded'] and t1['p10_pct']>base['p10_pct'] and t1['worst_absolute_increase']<=base['worst_absolute_increase']
  gates.append(dict(seed=seed,gain_better_than_T0=gain,harm_lower_than_T0=harm,explanatory_metric_improved=explain,mechanism_pass=mech,online_gate_pass=usable))
 monitor=[]
 for seed in [17,29]:
  for group in ['T0','T1']:
   for epoch in [-1,0,1]:
    zz=[r for r in info if r['seed']==seed and r['group']==group and r['epoch']==epoch and r['split']=='train_monitor'];monitor.append(dict(seed=seed,group=group,epoch=epoch,n=len(zz),mean_margin_loss=float(np.mean([r['margin_loss'] for r in zz])),median_margin_loss=float(np.median([r['margin_loss'] for r in zz]))))
 table('train_monitor_summary.csv',monitor)
 losses=[]
 for seed in [17,29]:
  for group in ['T0','T1']:
   for epoch in [0,1]:
    zz=[x for x in curves if x['seed']==seed and x['group']==group and x['epoch']==epoch];row=dict(seed=seed,group=group,epoch=epoch,n=len(zz))
    for k in ['local_loss','quality_loss','margin_loss','set_loss','gradnorm']:row[k+'_mean']=float(np.mean([x[k] for x in zz]));row[k+'_median']=float(np.median([x[k] for x in zz]))
    losses.append(row)
 table('training_loss_summary.csv',losses)
 for name,rows in [('decomposition.csv',decomposition),('decomposition_summary.csv',ds),('decision_per_event.csv',allrows),('decision_summary.csv',summary),('conditional_summary.csv',condition),('fixed_pair_conditional.csv',fixedpairs),('training_curves.csv',curves),('information_quality.csv',info)]:table(name,clean(rows))
 dump(DEST/'gates.json',dict(per_seed=gates,mechanism_pass=all(x['mechanism_pass'] for x in gates),online_pass=all(x['online_gate_pass'] for x in gates),test_accessed=False));print(json.dumps(gates,indent=2),flush=True)
if __name__=='__main__':report()
