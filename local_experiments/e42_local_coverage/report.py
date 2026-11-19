"""Fixed final-checkpoint comparison; no development checkpoint selection."""
from .core import *
from ..e41_training_signal.report import decision_summary

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def run():
 decisions=[];info=[];curves=[]
 for seed in [17,29]:
  for r in read(E41/f's{seed}/decisions.csv'):r['arm']='E29';decisions.append(r)
  for arm in ['T0','T1']:
   dest=D/f'{arm}_s{seed}';assert (dest/'evaluation_complete.json').exists();decisions+=read(dest/'decisions.csv');info+=read(dest/'information.csv')
   source=E39/f'T0_s{seed}/curves.json' if arm=='T0' else dest/'curves.json'
   meta={x['event']:x for x in json.loads((CACHE/f'train_s{seed}/metadata.json').read_text())}
   for r in json.loads(source.read_text()):curves.append(dict(**r,seed=seed,arm=arm,layer=meta[r['event']]['layer'],head=meta[r['event']]['head'],trajectory=meta[r['event']]['trajectory']))
 summaries=[]
 for dimension in ['overall','trajectory','layer','head']:
  gs=collections.defaultdict(list)
  for r in decisions:gs[(r['seed'],r['split'],r['arm'],r['policy'],r.get(dimension,'all') if dimension!='overall' else 'all')].append(r)
  for (seed,split,arm,policy,value),rs in gs.items():
   result=decision_summary(rs);harm=sorted([max(0.,-float(r['executed_gain'])) for r in rs],reverse=True);total=sum(harm);result.update(max1_harm_share=harm[0]/total if total else 0.,max5_harm_share=sum(harm[:5])/total if total else 0.)
   for L in [1,4,8]:
    vals=[float(r['top'+str(L)]) for r in rs if 'top'+str(L) in r]
    result['top%d_recovery_pct'%L]=100*sum(vals)/result['oracle_gain'] if vals and result['oracle_gain'] else None
   summaries.append(dict(seed=seed,split=split,arm=arm,policy=policy,dimension=dimension,value=value,**result))
 table(D/'decision_summary.csv',summaries)
 ir=[]
 for dimension in ['overall','trajectory','layer','head']:
  gs=collections.defaultdict(list)
  for r in info:gs[(r['seed'],r['split'],r['arm'],r['role'],r['status'],r.get(dimension,'all') if dimension!='overall' else 'all')].append(r)
  for (seed,split,arm,role,status,value),rs in gs.items():
   row=dict(seed=seed,split=split,arm=arm,role=role,status=status,dimension=dimension,value=value,events=len(rs))
   for k in ['local_mse','true_mass_weighted_local_mse','summary_weighted_local_sum','beta_mae','beta_bias','predicted_mass_share','true_mass_share','mass_l1']:
    v=np.array([float(r[k]) for r in rs]);row[k+'_mean']=float(v.mean());row[k+'_median']=float(np.median(v))
   ir.append(row)
 table(D/'information_summary.csv',ir)
 cr=[]
 for dimension in ['overall','layer','head']:
  gs=collections.defaultdict(list)
  for r in curves:gs[(r['seed'],r['arm'],r['epoch'],r.get(dimension,'all') if dimension!='overall' else 'all')].append(r)
  for (seed,arm,epoch,value),rs in gs.items():
   row=dict(seed=seed,arm=arm,epoch=epoch,dimension=dimension,value=value,updates=len(rs))
   for k in ['local_loss','quality_loss','set_loss','gradnorm']:
    x=np.array([float(r[k]) for r in rs]);row[k+'_mean']=float(x.mean());row[k+'_median']=float(np.median(x));row[k+'_p90']=float(np.percentile(x,90));row[k+'_max']=float(x.max())
   row['clip_fraction']=float(np.mean([r['gradnorm']>1 for r in rs]));cr.append(row)
 table(D/'training_gradient_summary.csv',cr)
 headline=[r for r in summaries if r['dimension']=='overall'];dump(D/'headline.json',headline)
 gates=[]
 for seed in ['17','29']:
  def get(arm,split,policy='Set'):return next(r for r in headline if str(r['seed'])==seed and r['arm']==arm and r['split']==split and r['policy']==policy)
  tr0,tr1=get('T0','train'),get('T1','train');dv0,dv1=get('T0','development'),get('T1','development');old=get('E29','development');ind=get('T1','development','Independent');traj=[r for r in summaries if str(r['seed'])==seed and r['arm']=='T1' and r['split']=='development' and r['policy']=='Set' and r['dimension']=='trajectory']
  checks=dict(train_gain_up=tr1['gain']>tr0['gain'],train_harm_down=tr1['H_loss']<tr0['H_loss'],dev_gain_up=dv1['gain']>dv0['gain']);mechanism=all(checks.values())
  deployment=dict(recovery20=dv1['recovery_pct']>=20,Set_beats_Independent=dv1['gain']>ind['gain'],Set_beats_noop=dv1['gain']>0,positive_dev_trajectories=all(r['gain']>0 for r in traj),nonnegative_mean=dv1['mean_relative_pct']>=0,harm_count_improves=dv1['harm']<old['harm'],p10_improves=dv1['p10_relative_pct']>old['p10_relative_pct'],worst_improves=dv1['worst_absolute_increase']<old['worst_absolute_increase'])
  deployment={k:bool(v) for k,v in deployment.items()}
  gates.append(dict(seed=int(seed),mechanism=mechanism,checks=checks,deployment=all(deployment.values()),deployment_checks=deployment))
 dump(D/'final_gates.json',dict(coverage_preflight=json.loads((D/'preflight_gate.json').read_text())['passed'],mechanism=all(r['mechanism'] for r in gates),deployment=all(r['deployment'] for r in gates),seeds=gates,test='sealed',further_training=False))
 print(json.dumps(headline,indent=2));print('GATES',json.dumps(gates))
if __name__=='__main__':run()
