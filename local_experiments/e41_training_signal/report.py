"""Equal-event summaries plus pooled decision gains; never pool gradient units blindly."""
from .run import D,dump,table
import csv,collections,json
import numpy as np

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def val(r,k):return float(r[k])
def decision_summary(rows):
 ini=np.array([val(r,'initial') for r in rows]);g=np.array([val(r,'executed_gain') for r in rows]);o=sum(val(r,'oracle_gain') for r in rows);h=sum(val(r,'hindsight_gain') for r in rows);rel=100*g[ini>1e-12]/ini[ini>1e-12]
 return dict(events=len(rows),initial_sum=float(ini.sum()),gain=float(g.sum()),oracle_gain=o,recovery_pct=100*g.sum()/o if o else None,hindsight_recovery_pct=100*h/o if o else None,pooled_improvement_pct=100*g.sum()/ini.sum(),mean_relative_pct=float(rel.mean()),p10_relative_pct=float(np.percentile(rel,10)),harm=int((g< -1e-12).sum()),positive=int((g>1e-12).sum()),noop=sum(r['execute']=='False' for r in rows),H_gain=float(g.clip(0).sum()),H_loss=float((-g).clip(0).sum()),worst_absolute_increase=float(max(0.,-g.min())))
def run():
 decisions=[];roles=[];pg=[]
 for seed in [17,29]:
  assert (D/f's{seed}/complete.json').exists()
  decisions+=read(D/f's{seed}/decisions.csv');roles+=read(D/f's{seed}/output_gradient_roles.csv');pg+=read(D/f's{seed}/parameter_gradients.csv')
 out=[]
 for dim in ['overall','trajectory','layer','head']:
  groups=collections.defaultdict(list)
  for r in decisions:groups[(r['seed'],r['split'],r['policy'],r.get(dim,'all') if dim!='overall' else 'all')].append(r)
  for (seed,split,policy,name),rs in groups.items():out.append(dict(seed=seed,split=split,policy=policy,dimension=dim,value=name,**decision_summary(rs)))
 table(D/'decision_summary.csv',out)
 rout=[]
 for dim in ['overall','trajectory','layer','head']:
  groups=collections.defaultdict(list)
  for r in roles:groups[(r['seed'],r['split'],r['role'],r.get(dim,'all') if dim!='overall' else 'all')].append(r)
  metrics=[k for k in roles[0] if k.endswith('_share') or k in ['page_fraction','beta_mae','beta_bias','beta_interval_outside_fraction','interval_excess_mean','beta_scale_mean','under_summary_true_mass_fraction','under_predicted_true_mass_fraction','beta_gradient_correct_fraction','set_beta_gradient_correct_fraction']]
  for (seed,split,role,name),rs in groups.items():
   row=dict(seed=seed,split=split,role=role,dimension=dim,value=name,events=len(rs))
   for k in metrics:
    z=[float(r[k]) for r in rs if r[k] and np.isfinite(float(r[k]))]
    row[k+'_mean']=float(np.mean(z)) if z else None;row[k+'_median']=float(np.median(z)) if z else None
   rout.append(row)
 table(D/'gradient_role_summary.csv',rout)
 gout=[]
 for dim in ['overall','trajectory','layer','head']:
  groups=collections.defaultdict(list)
  for r in pg:groups[(r['seed'],r['parameter_group'],r.get(dim,'all') if dim!='overall' else 'all')].append(r)
  metrics=[k for k in pg[0] if k.endswith('_norm') or k.startswith('cos_') or k.endswith('_projection') or k in ['clip_scale','cancellation_ratio']]
  for (seed,group,name),rs in groups.items():
   row=dict(seed=seed,parameter_group=group,dimension=dim,value=name,events=len(rs))
   for k in metrics:
    z=[float(r[k]) for r in rs if r[k]];row[k+'_median']=float(np.median(z)) if z else None
    if k.startswith('cos_'):row[k+'_negative_fraction']=float(np.mean(np.array(z)<0)) if z else None
   row['clipped_fraction']=float(np.mean([float(r['clip_scale'])<1 for r in rs]));gout.append(row)
 table(D/'parameter_gradient_summary.csv',gout)
 # Describe gradient magnitude versus target scale within events. This is not a new loss.
 headline=[r for r in out if r['dimension']=='overall'];dump(D/'headline.json',headline)
 print(json.dumps(headline,indent=2));print('PARAMETER',json.dumps([r for r in gout if r['dimension']=='overall'],indent=2))
if __name__=='__main__':run()
