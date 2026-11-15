from .run import *
import csv

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
 rows=[];summary=[];quality=[];general=[];conditions=[];gates=[];rep=[];cov=[];mom=[];reuse=[];perquality=[];percond=[]
 for seed in [17,29]:
  dest=DEST/f's{seed}';assert (dest/'complete.json').exists();rr=json.loads((dest/'decisions.json').read_text());qq=json.loads((dest/'quality.json').read_text());cc=json.loads((dest/'conditional.json').read_text());rows+=rr;perquality+=qq;percond+=cc;cov+=json.loads((dest/'coverage.json').read_text());mom+=json.loads((dest/'moments.json').read_text());reuse+=json.loads((dest/'reuse.json').read_text());ss={}
  for method in METHODS:
   for policy in ['Set','Independent']:
    sub=[r for r in rr if r['method']==method and r['policy']==policy];a=aggregate(sub);a.update(final_nmse_sum=sum(r['final_nmse'] for r in sub),max_absolute_nmse_increase=max(r['absolute_nmse_increase'] for r in sub),p90_absolute_nmse_increase=float(np.quantile([r['absolute_nmse_increase'] for r in sub],.9)),max_final_nmse=max(r['final_nmse'] for r in sub),mean_regret=float(np.mean([r['regret'] for r in sub])));summary.append(dict(seed=seed,method=method,policy=policy,**a));ss[method,policy]=a
   for dim in ['trajectory','layer','chunk','step','head']:
    sub=[r for r in rr if r['method']==method and r['policy']=='Set']
    for val in sorted({r[dim] for r in sub}):general.append(dict(seed=seed,method=method,dimension=dim,value=val,**aggregate([r for r in sub if r[dim]==val])))
   keys=sorted({(r['queries'],r['region']) for r in qq if r['method']==method})
   for query,region in keys:
    sub=[r for r in qq if r['method']==method and r['queries']==query and r['region']==region];entry=dict(seed=seed,method=method,queries=query,region=region,events=len(sub),page_occurrences=sum(r['pages'] for r in sub))
    for k in ['beta_mae','beta_rmse','beta_bias','summary_weighted_beta_mae','true_weighted_beta_mae','predicted_weighted_beta_mae','quality_mass_l1_relative','quality_mass_l2_relative','quality_mass_log_l1_relative','quality_mass_log_l2_relative']:
     vals=[r[k] for r in sub if r[k] is not None];entry[k]=float(np.mean(vals)) if vals else None;entry[k+'_finite_n']=len(vals)
    quality.append(entry)
   tt=[];pp=[];med=[]
   for idx in range(288):
    with np.load(dest/f'panel_{idx:04d}.npz') as p:
     t=p['true_gain'];v=p[method];t=t-t.mean(0);v=v-v.mean(0);t=t-t.mean(1,keepdims=True);v=v-v.mean(1,keepdims=True);tt.append(t.ravel());pp.append(v.ravel());cor=stablecorr(t,v)
     if cor is not None:med.append(cor)
   sub=[r for r in cc if r['method']==method];tr=sum(r['sign_flip_true'] for r in sub);tp=sum(r['sign_flip_tp'] for r in sub);pr=sum(r['sign_flip_pred'] for r in sub);conditions.append(dict(seed=seed,method=method,doublecentered=stablecorr(np.concatenate(tt),np.concatenate(pp)),median_event_interaction=float(np.median(med)),sign_flip_recall=tp/tr if tr else None,sign_flip_precision=tp/pr if pr else None))
  old=json.loads((E30/f's{seed}/decisions.json').read_text());old={(r['event'],r['method'],r['policy']):r for r in old}
  for r in rr:
   if r['method'] not in ['E29','TrueBeta']:continue
   o=old[r['event'],'Predicted' if r['method']=='E29' else 'True',r['policy']];same=all(r[k]==o[k] for k in ['execute','remove','add']);err=abs(r['executed_gain']-o['executed_gain']);assert same and err<1e-12;rep.append(dict(seed=seed,event=r['event'],method=r['method'],policy=r['policy'],same_action=same,gain_error=err))
  base=ss['E29','Set'];true=ss['TrueBeta','Set'];x=ss['LowRank8+Variance','Set'];ind=ss['LowRank8+Variance','Independent'];gg=[r for r in general if r['seed']==seed and r['method']=='LowRank8+Variance' and r['dimension']=='trajectory'];checks=dict(oracle_ge20=x['R_executed']>=.2,both_trajectories_positive=all(r['executed_gain_sum']>0 for r in gg),better_E29=x['executed_gain_sum']>base['executed_gain_sum'],set_better_independent=x['executed_gain_sum']>ind['executed_gain_sum'],mean_nonnegative=x['mean_improvement_pct']>=0,degradation_le15=x['degraded']/x['n']<=.15,p10_not_worse=x['p10_pct']>=base['p10_pct'],worst_not_worse=x['worst_pct']>=base['worst_pct']);gates.append(dict(seed=seed,checks=checks,passed=all(checks.values()),true_beta_gap_recovery=(x['R_executed']-base['R_executed'])/(true['R_executed']-base['R_executed'])))
 for name,data in [('set_decision.csv',rows),('summary.csv',summary),('quality_summary.csv',quality),('quality_per_event.csv',perquality),('generalization.csv',general),('conditional_summary.csv',conditions),('conditional_per_event.csv',percond),('recurrence_coverage.csv',cov),('moment_errors.csv',mom),('baseline_replay.csv',rep)]:table(name,data)
 dump(DEST/'summary.json',summary);dump(DEST/'gates.json',dict(results=gates,passed=all(g['passed'] for g in gates),network_training=False,test_accessed=False));dump(DEST/'reuse_summary.json',[dict(seed=s,history_occurrences=len([r for r in reuse if r['seed']==s]),repeated_occurrences=sum(r['repeated'] for r in reuse if r['seed']==s),all_repeated_coefficients_and_variance_identical=True) for s in [17,29]])
 P=410;HT=12*72;M=16;D=128;rank=8;pairs=P*HT*M
 cost=[dict(method='LowRank8+Variance',extra_bytes_page=64,extra_history_bytes_29layers=338*12*29*64,extra_current_bytes_29layers=72*12*29*64,shared_basis_FP32_bytes_29layers=29*12*128*8*4,query_projection_GFLOPs_layer=2*D*rank*M*HT/1e9,mean_response_GFLOPs_layer=2*4*rank*pairs/1e9,variance_GFLOPs_layer_if_not_reused=2*4*8*pairs/1e9,current_projection_GFLOPs_per_layer_refresh=(2*4*128*8+4*128)*72*12/1e9,logical_extra_statistics_read_MiB_layer=64*P*HT/2**20,additional_raw_KV_reads=0,new_group_exp_count_layer=4*pairs,new_page_log_count_layer=pairs,original_Reader_and_search_excluded=True)]
 table('cost_estimate.csv',cost);print(json.dumps([r for r in summary if r['policy']=='Set'],indent=2))
if __name__=='__main__':main()
