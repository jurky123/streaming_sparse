from .core import *
from scipy.stats import rankdata

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
def auc(rows,feature):
 rr=[r for r in rows if abs(r['true_proposal_gain'])>1e-12];y=np.array([r['true_proposal_gain']>0 for r in rr]);n=int(y.sum());m=len(y)-n
 if not n or not m:return None
 ranks=rankdata([r[feature] for r in rr]);return float((ranks[y].sum()-n*(n+1)/2)/(n*m))
def stats(rows,fields):
 out=dict(n=len(rows))
 for name in fields:
  v=[r[name] for r in rows];out[name+'_sum']=sum(v);out[name+'_mean']=float(np.mean(v)) if v else None;out[name+'_median']=float(np.median(v)) if v else None
 return out

def reportB():
 allrows={k:[] for k in ['action_margin_attribution','region_attribution','endpoint_rescue','teacher_residual_ladder','acceptance_features']};bs=[];rs=[];es=[];ls=[];front=[];gates=[]
 for seed in [17,29]:
  dest=DEST/f'B_s{seed}';assert (dest/'complete.json').exists();data={k:json.loads((dest/(k+'.json')).read_text()) for k in allrows}
  for k,v in data.items():allrows[k]+=v
  for dimension in ['ALL','layer','trajectory']:
   values=['ALL'] if dimension=='ALL' else sorted({r[dimension] for r in data['action_margin_attribution']})
   for v in values:
    select=lambda name:[r for r in data[name] if dimension=='ALL' or r[dimension]==v]
    rr=select('action_margin_attribution');wrong=[r for r in rr if not r['same_action']]
    for subset,z in [('all',rr),('different_action',wrong)]:bs.append(dict(seed=seed,dimension=dimension,value=v,subset=subset,**stats(z,['flip_error','phi_response','phi_beta','interaction','remaining_teacher_error']),same_action=sum(r['same_action'] for r in z),beta_positive=sum(r['phi_beta']>0 for r in z),beta_exceeds_response=sum(r['phi_beta']>r['phi_response'] for r in z),M10_preference_correct=sum(r['M10']<=0 for r in z),M01_preference_correct=sum(r['M01']<=0 for r in z),M11_preference_correct=sum(r['M11']<=0 for r in z)))
    zz=select('region_attribution');rs.append(dict(seed=seed,dimension=dimension,value=v,**stats(zz,['phi_endpoint','phi_other','phi_background']),mean_endpoint_pages=float(np.mean([r['end_pages'] for r in zz]))))
    for mode in ['beta','response','both']:
     zz=[r for r in select('endpoint_rescue') if r['mode']==mode];es.append(dict(seed=seed,dimension=dimension,value=v,mode=mode,n=len(zz),original_rejected=sum(r['original_action_rejected'] for r in zz),**({result:sum(r['result']==result for r in zz) for result in ['noop','positive','harmful','near_zero']} if mode=='both' else {}),**({'old_gain_sum':sum(r['old_gain'] for r in zz),'new_gain_sum':sum(r['new_gain'] for r in zz)} if mode=='both' else {})))
    zz=select('teacher_residual_ladder')
    for stage in ['pred_M16','true_M16_B16','true_Q64_B16','true_Q64_full','true_Q64_perp']:
     margins=[r[stage+'_wrong']-r[stage+'_oracle'] for r in zz];ls.append(dict(seed=seed,dimension=dimension,value=v,stage=stage,n=len(zz),margin_sum=sum(margins),margin_mean=float(np.mean(margins)) if zz else None,wrong_action_nonpositive=sum(r[stage+'_wrong']<=0 for r in zz),fixed_preference_correct=sum(m<=0 for m in margins)))
  wrong=[r for r in data['action_margin_attribution'] if not r['same_action']];er=[r for r in data['endpoint_rescue'] if r['mode']=='beta'];bp=np.mean([r['phi_beta'] for r in wrong]);rp=np.mean([r['phi_response'] for r in wrong]);rescue=sum(r['original_action_rejected'] for r in er)/len(er) if er else 0.;cpass=bool(bp>0 and bp>rp and rescue>=.25)
  ex=[r for r in data['acceptance_features'] if r['executed']];dpass=True;overall={}
  edges=np.quantile([r['gain'] for r in ex],[0,1/3,2/3,1]);bins=np.searchsorted(edges[1:-1],[r['gain'] for r in ex],side='right')
  for feature in ['rho','pi_positive']:
   vv=[]
   for k in range(3):
    group=[r for r,b in zip(ex,bins) if b==k];aa=auc(group,feature);positive=sum(r['true_proposal_gain']>1e-12 for r in group);negative=sum(r['true_proposal_gain']< -1e-12 for r in group);valid=positive>=5 and negative>=5
    front.append(dict(seed=seed,scope='gain_tertile',value=k,feature=feature,n=len(group),positive=positive,harmful=negative,auc=aa,gate_eligible=valid))
    if valid and aa is not None:vv.append(aa)
   overall[feature]=float(np.median(vv)) if vv else 0.
  for traj in sorted({r['trajectory'] for r in ex}):
   rr=[r for r in ex if r['trajectory']==traj];ok=False
   for feature in ['rho','pi_positive']:
    aa=auc(rr,feature);front.append(dict(seed=seed,scope='trajectory',value=traj,feature=feature,n=len(rr),auc=aa,median_conditional_auc=overall[feature]));ok|=aa is not None and aa>=.6 and overall[feature]>=.6
   dpass&=ok
  for layer in [1,15,29]:
   rr=[r for r in ex if r['layer']==layer]
   for feature in ['rho','pi_positive']:front.append(dict(seed=seed,scope='layer',value=layer,feature=feature,n=len(rr),auc=auc(rr,feature)))
  gates.append(dict(seed=seed,C_beta_mean=float(bp),C_response_mean=float(rp),C_endpoint_beta_rejection=rescue,C_pass=cpass,D_pass=bool(dpass),D_gain_conditional_auc=overall))
 for k,v in allrows.items():table(k+'.csv',v)
 table('action_margin_summary.csv',bs);table('region_summary.csv',rs);table('endpoint_rescue_summary.csv',es);table('teacher_ladder_summary.csv',ls);table('acceptance_frontier.csv',front);dump(DEST/'conditional_gates.json',dict(C_pass=all(x['C_pass'] for x in gates),D_pass=all(x['D_pass'] for x in gates),seeds=gates));print(json.dumps(gates,indent=2))

def reportA():
 rows=[];conds=[];summary=[];general=[];qualities=[];weights=[];cs=[]
 for seed in [17,29]:
  dest=DEST/f'A_s{seed}';assert (dest/'complete.json').exists();dr=json.loads((dest/'decisions.json').read_text());cr=json.loads((dest/'conditional.json').read_text());qr=json.loads((dest/'quality.json').read_text());ww=json.loads((dest/'weights.json').read_text());rows+=dr;conds+=cr;qualities+=qr;weights+=ww
  for method in METHODS:
   for policy in ['Set','Independent']:
    rr=[r for r in dr if r['method']==method and r['policy']==policy];summary.append(dict(seed=seed,method=method,policy=policy,**aggregate(rr),worst_absolute_nmse_increase=max(-r['executed_gain'] for r in rr),fixed_preference_correct=sum(r['fixed_margin']<=0 for r in rr)))
    for dim in ['layer','trajectory','chunk','step','head']:
     for val in sorted({r[dim] for r in rr}):general.append(dict(seed=seed,method=method,policy=policy,dimension=dim,value=val,**aggregate([r for r in rr if r[dim]==val])))
   cc=[r for r in cr if r['method']==method];tc=[];pc=[];dt=[];dp=[]
   for idx in range(288):
    p=np.load(dest/f'panel_{idx:04d}.npz');t=p['true_gain'];v=p[method];tt=t-t.mean(0);pp=v-v.mean(0);tc.extend(tt.ravel());pc.extend(pp.ravel());dt.extend((tt-tt.mean(1,keepdims=True)).ravel());dp.extend((pp-pp.mean(1,keepdims=True)).ravel())
   out=dict(seed=seed,method=method,conditional_raw=stablecorr(tc,pc),double_centered=stablecorr(dt,dp),median_event_corr=float(np.median([r['conditional_corr'] for r in cc if r['conditional_corr'] is not None])))
   for kind in ['sign_flip','rank_flip']:
    n=sum(r[kind+'_true'] for r in cc);out[kind+'_recall']=sum(r[kind+'_tp'] for r in cc)/n if n else None
   cs.append(out)
 table('role_weight_ablation.csv',summary);table('role_weight_per_event.csv',rows);table('generalization.csv',general);table('quality_per_event.csv',qualities);table('weight_concentration.csv',weights);table('conditional_summary.csv',cs);table('conditional_per_event.csv',conds)
 # Group/query-normalized weights sum to1, so concentration is reported by role/layer rather than misleading raw magnitude pooling.
 ws=[]
 for seed in [17,29]:
  for m in METHODS[2:]:
   for layer in [1,15,29]:
    for group in range(2 if m.startswith('G2') else 4):
     z=[r for r in weights if r['seed']==seed and r['method']==m and r['layer']==layer and r['state']==0 and r['group']==group];ws.append(dict(seed=seed,method=m,layer=layer,group=group,n=len(z),mean_max_normalized_weight=float(np.mean([r['max_weight'] for r in z])),mean_effective_pages=float(np.mean([r['effective_pages'] for r in z])),mean_abs_offset=float(np.mean([abs(r['offset_mean']) for r in z]))))
 table('weight_summary.csv',ws)
 print('A tables saved')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['A','B'],required=True);a=p.parse_args();{'A':reportA,'B':reportB}[a.stage]()
