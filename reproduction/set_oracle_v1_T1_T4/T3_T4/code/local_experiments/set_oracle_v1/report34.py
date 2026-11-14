import json,csv,time
from collections import defaultdict
import numpy as np
from .stage34 import ROOT,OUT,split
from .common import dump,sha

def csvwrite(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def aggregate(rows):
 out=[]
 for (sp,method,step),group in groupby(rows,lambda r:(r['split'],r['method'],r['step'])).items():
  gain=np.array([r['gain_vs_risk'] for r in group]);recover=[r['recovery_of_full_4step'] for r in group if r['recovery_of_full_4step'] is not None]
  out.append(dict(split=sp,method=method,step=step,n=len(group),mean_gain_pct=float(gain.mean()*100),median_gain_pct=float(np.median(gain)*100),p10_gain_pct=float(np.quantile(gain,.1)*100),p90_gain_pct=float(np.quantile(gain,.9)*100),improved=int((gain>1e-9).sum()),worse=int((gain<-1e-9).sum()),mean_full4_recovery=float(np.mean(recover)) if recover else None,median_full4_recovery=float(np.median(recover)) if recover else None))
 return out
def groupby(rows,fn):
 d=defaultdict(list)
 for r in rows:d[fn(r)].append(r)
 return d

def run():
 t3files=sorted((OUT/'t3').glob('*.json'));t4files=sorted((OUT/'t4').glob('*.json'));assert len(t3files)==len(t4files)==144
 t3={p.stem:json.loads(p.read_text()) for p in t3files};t4={p.stem:json.loads(p.read_text()) for p in t4files}
 for eid,d in t3.items():
  paired=json.loads((OUT/'t3_paired'/f'{eid}.json').read_text());d['results'].update(paired['results'])
 validation=defaultdict(list)
 for eid,data in t4.items():
  if data['split']!='validation':continue
  for method,result in data['results'].items():
   for step in (1,2,4):
    trace=result['trace'];validation[(method,step)].append((trace[0]['nmse']-trace[step]['nmse'])/trace[0]['nmse'])
 chosen=max(validation,key=lambda key:(np.mean(validation[key]),-key[1],key[0]))
 selected=dict(method=chosen[0],steps=chosen[1],validation_mean_gain_pct=float(np.mean(validation[chosen])*100),selection='maximize mean relative NMSE reduction on 36 city_walk seed303 instances, no evaluation labels',trials=len(validation))
 dump(OUT/'selected_model.json',selected)
 for stage,all_data in [('t3',t3),('t4',t4)]:
  rows=[];actions=[]
  for eid,data in all_data.items():
   base=t3[eid]['results']['full_q64_full']['trace'];fullgain=base[0]['nmse']-base[4]['nmse']
   for method,result in data['results'].items():
    for step in (1,2,4):
     j=result['trace'][step]['nmse'];gain=base[0]['nmse']-j
     rows.append(dict(event_id=eid,split=data['split'],method=method,step=step,risk_nmse=base[0]['nmse'],nmse=j,gain_vs_risk=gain/base[0]['nmse'],recovery_of_full_4step=gain/fullgain if fullgain>1e-12 else None))
    for d in result['diagnostics']:actions.append(dict(event_id=eid,split=data['split'],method=method,**d))
  csvwrite(OUT/f'{stage}_per_instance.csv',rows);agg=aggregate(rows);csvwrite(OUT/f'{stage}_summary.csv',agg);dump(OUT/f'{stage}_summary.json',agg);csvwrite(OUT/f'{stage}_actions.csv',actions)
  action_summary=[]
  for (sp,m),g in groupby([a for a in actions if a['step']==1],lambda a:(a['split'],a['method'])).items():
   ratios=[a['pool_gain_ratio'] for a in g if a['pool_gain_ratio'] is not None];pos=[a for a in g if a['best_full_gain']>1e-12];corr=[a['spearman'] for a in g if a['spearman'] is not None]
   action_summary.append(dict(split=sp,method=m,n=len(g),positive_full=len(pos),best_in_pool_rate=float(np.mean([a['best_in_pool'] for a in pos])) if pos else None,best_action_hit_rate=float(np.mean([a['best_action_hit'] for a in pos])) if pos else None,median_pool_gain_ratio=float(np.median(ratios)) if ratios else None,mean_pool_gain_ratio=float(np.mean(ratios)) if ratios else None,false_positive_rate=float(np.mean([a['false_positive'] for a in g])),mean_spearman=float(np.mean(corr)) if corr else None,actions=g[0]['actions']))
  csvwrite(OUT/f'{stage}_first_action_summary.csv',action_summary);dump(OUT/f'{stage}_first_action_summary.json',action_summary)
 chosenrows=[]
 for eid,d in t4.items():
  r=d['results'][chosen[0]];base=r['trace'][0]['nmse'];j=r['trace'][chosen[1]]['nmse'];chosenrows.append(dict(event_id=eid,split=d['split'],risk_nmse=base,nmse=j,gain_pct=100*(base-j)/base,feature_predict_ms=1000*r['feature_predict_seconds'],objective_build_ms=1000*r['objective_build_seconds'],four_step_decision_ms=1000*r['decision_seconds_median'],clipped_logratio_count=r['clipped_logratio_count']))
 csvwrite(OUT/'t4_selected_per_instance.csv',chosenrows)
 print(json.dumps(selected))
 for sp,g in groupby(chosenrows,lambda a:a['split']).items():print(sp,'gainmean',np.mean([a['gain_pct'] for a in g]),'median',np.median([a['gain_pct'] for a in g]),'worse',sum(a['gain_pct']<-1e-7 for a in g))
 dump(OUT/'completion.json',dict(T3=144,T4=144,selected=selected,source_sha256=sha(__file__),finished=time.time()))
if __name__=='__main__':run()
