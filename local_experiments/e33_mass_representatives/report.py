from .run import *
import csv

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def agg(rows):
 old=sum(r['old_risk'] for r in rows);new=sum(r['new_risk'] for r in rows);same=sum(r['same_mask_nmse'] for r in rows);oracle=sum(r['new_oracle'] for r in rows);n=sum(r['covered_pages']*64 for r in rows);rmse=math.sqrt(sum((r['beta_mse'] or 0)*r['covered_pages']*64 for r in rows)/n) if n else None;brmse=math.sqrt(sum((r['baseline_beta_mse'] or 0)*r['covered_pages']*64 for r in rows)/n) if n else None
 rel=[100*(r['old_risk']-r['new_risk'])/r['old_risk'] for r in rows if r['old_risk']>1e-12]
 return dict(events=len(rows),covered_events=sum(r['covered_pages']>0 for r in rows),covered_page_occurrences=n//64,history_mass_fraction_mean=float(np.mean([r['history_mass_fraction'] for r in rows])),old_risk_sum=old,same_mask_sum=same,new_risk_sum=new,new_oracle_sum=oracle,same_mask_improvement_pct=100*(old-same)/old,new_risk_improvement_pct=100*(old-new)/old,new_oracle_vs_old_improvement_pct=100*(old-oracle)/old,oracle_gain_sum=new-oracle,oracle_gain_relative_new=1-oracle/new,beta_rmse=rmse,baseline_beta_rmse=brmse,beta_rmse_reduction=1-rmse/brmse if brmse else None,beta_mae=sum((r['beta_mae'] or 0)*r['covered_pages']*64 for r in rows)/n if n else None,baseline_beta_mae=sum((r['baseline_beta_mae'] or 0)*r['covered_pages']*64 for r in rows)/n if n else None,beta_gt_log4_fraction=sum((r['beta_positive_gt_log4'] or 0)*r['covered_pages']*64 for r in rows)/n if n else None,quality_relative_l1_mean=float(np.mean([r['quality_relative_l1'] for r in rows if r['quality_relative_l1'] is not None])) if n else None,mean_relative_improvement_pct=float(np.mean(rel)),p10_pct=float(np.quantile(rel,.1)),worst_pct=min(rel),degraded=sum(r['new_risk']>r['old_risk']+1e-12 for r in rows),V_energy_sum=sum(r['V_energy'] for r in rows),Z_energy_sum=sum(r['Z_energy'] for r in rows),cross_energy_sum=sum(r['cross_energy'] for r in rows),net_nmse_sum=sum(r['net_nmse'] for r in rows))

def main():
 rows=[];fits=[];summ=[];general=[];gates=[]
 for seed in [17,29]:
  assert (DEST/f's{seed}/complete.json').exists();rr=json.loads((DEST/f's{seed}/rows.json').read_text());rows+=rr;fits+=json.loads((DEST/f's{seed}/fits.json').read_text())
  for name in sorted({r['method'] for r in rr}):
   ss=[r for r in rr if r['method']==name]
   for scope in ['all','covered']:
    selected=ss if scope=='all' else [r for r in ss if r['covered_pages']]
    if selected:summ.append(dict(seed=seed,method=name,scope=scope,**agg(selected)))
   for dim in ['trajectory','layer','chunk','step']:
    for value in sorted({r[dim] for r in ss}):
     sub=[r for r in ss if r[dim]==value and r['covered_pages']]
     if sub:general.append(dict(seed=seed,method=name,dimension=dim,value=value,**agg(sub)))
  for method in METHODS:
   x={k:next(r for r in summ if r['seed']==seed and r['method']==method+'-'+k and r['scope']=='covered') for k in KINDS};tr=[r for r in general if r['seed']==seed and r['method']==method+'-forward' and r['dimension']=='trajectory'];checks=dict(forward_rmse_10pct=x['forward']['beta_rmse_reduction']>=.1,forward_mixed_within2pct=x['forward']['new_risk_sum']<=1.02*x['forward']['old_risk_sum'],each_trajectory_within2pct=all(r['new_risk_sum']<=1.02*r['old_risk_sum'] for r in tr),tile_step_rmse_better=all(x[k]['beta_rmse_reduction']>0 for k in ['tile','step']),tile_step_mixed_within2pct=all(x[k]['new_risk_sum']<=1.02*x[k]['old_risk_sum'] for k in ['tile','step']));gates.append(dict(seed=seed,method=method,checks=checks,passed=all(checks.values())))
 table('per_event.csv',rows);table('summary.csv',summ);table('generalization.csv',general);table('fit_diagnostics.csv',fits);dump(DEST/'summary.json',summ);dump(DEST/'gates.json',dict(results=gates,passed=any(all(r['passed'] for r in gates if r['method']==m) for m in METHODS),B_started=False,test_accessed=False));print(json.dumps([r for r in summ if r['scope']=='covered'],indent=2))
if __name__=='__main__':main()
