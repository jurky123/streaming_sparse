"""Matched precision metrics for all fixed baselines and the feasible set."""
import json,csv
from pathlib import Path
import numpy as np
from .core import Problem
from ..e11_oracle.contributions import diagnostic,selected_check
root=Path('/tmp/e11_compact');rows=[]
for layer in (1,15,29):
 r=root/f'l{layer}';data=dict(np.load(f'/tmp/e11_results/contributions/horse_field_s101_c5_l{layer}_s0.npz'));p=Problem(data);loc=json.loads((r/'local_result.json').read_text());best=np.zeros(410,bool);best[loc['best_mask']]=True
 for name,mask in [('mass',data['mass']),('risk',data['risk']),('best_feasible',best)]:
  arrays,metrics=diagnostic(data,mask)
  ids=np.where(~mask)[0];rng=np.random.default_rng(17);first=rng.integers(0,len(ids),256);second=(first+rng.integers(1,len(ids),256))%len(ids);pairs=np.c_[ids[first],ids[second]];ea=p.e[:,pairs[:,0]];eb=p.e[:,pairs[:,1]];arrays['pair_cosines']=np.sum(ea*eb,axis=-1)/(np.linalg.norm(ea,axis=-1)*np.linalg.norm(eb,axis=-1));arrays['pair_indices_original_pages']=pairs;arrays['pair_indices']=np.c_[first,second];assert np.all(pairs[:,0]!=pairs[:,1]) and np.isfinite(arrays['pair_cosines']).all()
  fp=selected_check(Path(f'/tmp/e11_results/snapshots/horse_field_s101_c5_l{layer}_s0.pt'),mask,data);np.savez_compressed(r/f'{name}_full_D02.npz',**arrays);d=p.decomposition(mask)
  row=dict(layer=layer,head=int(data['head']),method=name,count=int(mask.sum()),own=bool(mask[p.own]),**metrics,**fp,A_nmse=d['A_nmse'],C_nmse=d['C_nmse'],mean_kappa=float(d['kappa'].mean()),negative_C_fraction=float(np.mean(d['C']<0)),mean_Z_ratio=float(d['denominator_ratio'].mean()),log_denominator_MAE=float(np.mean(np.abs(arrays['log_denominator_delta']))),fixed_dense_denominator_nmse=d['fixed_dense_denominator_nmse'],replaced_mass=int(np.sum(mask&~data['mass'])),replaced_risk=int(np.sum(mask&~data['risk'])),pair_cosine_mean=float(arrays['pair_cosines'].mean()),pair_cosine_q10=float(np.quantile(arrays['pair_cosines'],.1)),pair_cosine_q50=float(np.quantile(arrays['pair_cosines'],.5)),pair_cosine_q90=float(np.quantile(arrays['pair_cosines'],.9)))
  rows.append(row);print(layer,name,metrics['nmse'],flush=True)
  (r/f'{name}_mask.json').write_text(json.dumps(np.where(mask)[0].tolist()))
(root/'matched_metrics.json').write_text(json.dumps(rows,indent=2))
with (root/'matched_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Analysis-only counterfactual: keep the best residual but restore risk's denominator.
cf=[]
for layer in (1,15,29):
 risk=np.load(root/f'l{layer}/risk_full_D02.npz');best=np.load(root/f'l{layer}/best_feasible_full_D02.npz');metric=next(r for r in rows if r['layer']==layer and r['method']=='best_feasible');energy=best['Jq'].sum()/metric['nmse'];counter=float(np.sum(best['Jq']*(best['denominator_ratio']/risk['denominator_ratio'])**2)/energy);cf.append(dict(layer=layer,best_actual_nmse=metric['nmse'],best_residual_with_risk_denominator_nmse=counter,denominator_change_effect=metric['nmse']-counter,queries_with_larger_Z=int(np.sum(best['denominator_ratio']>=risk['denominator_ratio'])),analysis_only=True))
(root/'denominator_counterfactual.json').write_text(json.dumps(cf,indent=2))
