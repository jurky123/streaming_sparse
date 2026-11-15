from .run import *
import csv

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
 allrows=[];summary=[];replicates=[];general=[];stability=[];qualities=[];conditions=[];gates=[];replay=[]
 for seed in [17,29]:
  dest=DEST/f's{seed}';assert (dest/'complete.json').exists();rows=json.loads((dest/'decisions.json').read_text());allrows+=rows;qq=json.loads((dest/'quality.json').read_text());cc=json.loads((dest/'conditional.json').read_text());methods=BASES+[f'KeyProbe-b{b}' for b in BS];ss={}
  for method in methods:
   for policy in ['Set','Independent']:
    sub=[r for r in rows if r['method']==method and r['policy']==policy];ag=aggregate(sub);ag.update(seed=seed,method=method,policy=policy,unique_events=288,sampling_repetitions=5 if method.startswith('Key') else 1);summary.append(ag);ss[method,policy]=ag
    for rs in sorted({r['sampling_seed'] for r in sub},key=lambda x:-1 if x is None else x):
     replicates.append(dict(seed=seed,method=method,policy=policy,sampling_seed=rs,**aggregate([r for r in sub if r['sampling_seed']==rs])))
   sub=[r for r in rows if r['method']==method and r['policy']=='Set']
   for dim in ['trajectory','layer','chunk','step','head']:
    for v in sorted({r[dim] for r in sub}):general.append(dict(seed=seed,method=method,dimension=dim,value=v,**aggregate([r for r in sub if r[dim]==v])))
   qm=[r for r in qq if r['method']==method];quality=dict(seed=seed,method=method,n=len(qm))
   for k in ['beta_mae','beta_rmse','beta_signed_bias','gain_pearson','gain_spearman','positive_recall','sign_accuracy','summary_weighted_beta_mae','true_weighted_beta_mae','predicted_weighted_beta_mae','quality_mass_l1_relative','quality_mass_l2_relative']:
    values=[r[k] for r in qm if k in r and r[k] is not None];quality[k]=float(np.mean(values)) if values else None
   qualities.append(quality)
   dt=[];dp=[];med=[]
   for idx in range(288):
    with np.load(dest/f'panel_{idx:04d}.npz') as p:
     t=p['true_gain'];t=t-t.mean(0);t=t-t.mean(1,keepdims=True)
     keys=[method] if method in BASES else [f'{method}-r{s}' for s in SAMPLE_SEEDS]
     for key in keys:
      v=p[key];v=v-v.mean(0);v=v-v.mean(1,keepdims=True);dt.append(t.ravel());dp.append(v.ravel());cor=stablecorr(t,v)
      if cor is not None:med.append(cor)
   cm=[r for r in cc if r['method']==method];tr=sum(r['sign_flip_true'] for r in cm);tp=sum(r['sign_flip_tp'] for r in cm);pr=sum(r['sign_flip_pred'] for r in cm)
   conditions.append(dict(seed=seed,method=method,doublecentered=stablecorr(np.concatenate(dt),np.concatenate(dp)),median_event_interaction=float(np.median(med)),sign_flip_recall=tp/tr if tr else None,sign_flip_precision=tp/pr if pr else None))
   if method not in BASES:
    for event in sorted({r['event'] for r in sub}):
     ee=[r for r in sub if r['event']==event];actions=[(r['remove'],r['add']) if r['execute'] else ('noop',) for r in ee];counts=collections.Counter(actions);dis=sum(actions[i]!=actions[j] for i in range(5) for j in range(i+1,5))/10;stability.append(dict(seed=seed,method=method,event=event,distinct_actions=len(counts),modal_frequency=max(counts.values())/5,pair_disagreement=dis,positive_samples=sum(r['executed_gain']>1e-12 for r in ee),degraded_samples=sum(r['executed_gain']< -1e-12 for r in ee),gain_mean=float(np.mean([r['executed_gain'] for r in ee])),gain_std=float(np.std([r['executed_gain'] for r in ee]))))
  base=ss['E29','Set'];teacher=ss['Candidate-True','Set']
  for b in BS:
   method=f'KeyProbe-b{b}';x=ss[method,'Set'];ind=ss[method,'Independent'];gap=(x['R_executed']-base['R_executed'])/(teacher['R_executed']-base['R_executed']);rr=[r for r in replicates if r['seed']==seed and r['method']==method and r['policy']=='Set'];gg=[r for r in general if r['seed']==seed and r['method']==method and r['dimension']=='trajectory'];checks=dict(better_E29=x['R_executed']>base['R_executed'],better_independent=x['R_executed']>ind['R_executed'],candidate_gap_half=gap>=.5,each_trajectory_positive=all(r['executed_gain_sum']>0 for r in gg),four_of_five=sum(r['R_executed']>base['R_executed'] for r in rr)>=4);reliable=dict(mean_nonnegative=x['mean_improvement_pct']>=0,degradation_le15=x['degraded']/x['n']<=.15,p10_better=x['p10_pct']>base['p10_pct'],worst_better=x['worst_pct']>base['worst_pct']);gates.append(dict(seed=seed,b=b,candidate_gap_recovery=gap,information_checks=checks,information_pass=all(checks.values()),reliability_checks=reliable,reliability_pass=all(reliable.values())))
  old=json.loads((E30/f's{seed}'/'decisions.json').read_text());old={(r['event'],r['method'],r['policy']):r for r in old}
  for r in rows:
   if r['method'] not in BASES:continue
   om={'E29':'Predicted','Candidate-True':'Candidate-True','Full-True':'True'}[r['method']];o=old[r['event'],om,r['policy']];same=all(r[k]==o[k] for k in ['execute','remove','add']);error=abs(r['executed_gain']-o['executed_gain']);assert same and error<1e-12;replay.append(dict(seed=seed,event=r['event'],method=r['method'],policy=r['policy'],actions_equal=same,gain_error=error))
 table('quality_per_event.csv',[r for seed in [17,29] for r in json.loads((DEST/f's{seed}/quality.json').read_text())]);table('conditional_per_event.csv',[r for seed in [17,29] for r in json.loads((DEST/f's{seed}/conditional.json').read_text())])
 for name,data in [('set_decision.csv',allrows),('summary.csv',summary),('sampling_seed_summary.csv',replicates),('generalization.csv',general),('sampling_stability.csv',stability),('quality_summary.csv',qualities),('conditional_summary.csv',conditions),('baseline_replay.csv',replay)]:table(name,data)
 costs=[]
 for b in BS:
  tokens=48*4*b;byt=tokens*128*2
  for k,pages in [(21,216),(41,410)]:costs.append(dict(b=b,pages=pages,exact_pages=k,candidates=48,probes=16,key_tokens_per_headtile=tokens,logical_key_bytes_per_headtile=byt,extra_relative_exact_KV=byt/(k*64*128*2*2),logical_key_MiB_per_layer=byt*864/2**20,extra_QK_GFLOPs_layer=2*864*16*tokens*128/1e9,extra_AV_FLOPs=0,new_permanent_cache=0))
 table('cost_estimate.csv',costs);dump(DEST/'gates.json',dict(results=gates,information_pass=any(all(g['information_pass'] for g in gates if g['b']==b) for b in [2,4]),full_pass=any(all(g['information_pass'] and g['reliability_pass'] for g in gates if g['b']==b) for b in [2,4]),test_accessed=False,training=False))
 dump(DEST/'summary.json',summary);print(json.dumps([r for r in summary if r['policy']=='Set'],indent=2))
if __name__=='__main__':main()
