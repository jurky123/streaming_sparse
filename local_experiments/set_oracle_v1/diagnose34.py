import json,concurrent.futures
import numpy as np
from scipy.stats import spearmanr
from .stage34 import ROOT,OUT,split,online_load,online_features,Objective,rollout
from .problem import ResidualProblem
from .common import dump

def runone(eid):
 p=ResidualProblem(ROOT/'contributions'/eid);f=online_load(p.root/'summary_and_route_features.npz');pr=np.load(OUT/'projections.npz');x=p.data['risk'];r,a,truth=p.swaps(x);top=np.argsort(truth.ravel(),kind='stable')[:20];stats=[]
 for nq in (16,64):
  qi=f['probe_indices'].astype(int) if nq==16 else np.arange(64)
  for kind in ('random','pca'):
   for rank in (4,8,16,32):
    obj=Objective(p.e[qi]@pr[kind][:,:rank],p.d[qi],p.data['Zh'][qi],p.C);scores=obj.swaps(x,r,a).ravel();order=np.argsort(scores,kind='stable');ranks=np.empty(len(scores),int);ranks[order]=np.arange(len(scores));corr=spearmanr(np.arange(20),ranks[top]).statistic
    stats.append(dict(method=f'{kind}_r{rank}_q{nq}',top20_overlap=len(set(top)&set(order[:20]))/20,top20_rank_spearman=float(corr) if np.isfinite(corr) else None))
 selected=json.loads((OUT/'selected_model.json').read_text());method=selected['method'];lam=float(method.split('_')[0][5:]);nq=int(method.split('_')[1][1:]);pool=method.split('_')[-1];model=np.load(OUT/f'ridge_{lam}.npz');qi=f['probe_indices'].astype(int) if nq==16 else np.arange(64);fi=dict(f);fi['query']=f['query'][qi];fi['risk_features_q64']=f['risk_features_q64'][qi];features,mass=online_features(fi,model['feature']);pred=features@model['coef'];ep=pred[...,:16];dp=mass*np.expm1(np.clip(pred[...,-1],-20,20));scale=p.data['Zh'][qi].sum(1);et=(p.e[qi]@model['pca'])/scale[:,None,None];dt=p.d[qi]/scale[:,None]
 Epred=ep[:,~x].sum(1);Etrue=et[:,~x].sum(1);cos=np.sum(Epred*Etrue,axis=1)/(np.linalg.norm(Epred,axis=1)*np.linalg.norm(Etrue,axis=1)+1e-30);zp=1+dp[:,x].sum(1);zt=1+dt[:,x].sum(1)
 interventions={}
 for label,ee,dd in [('oracle_residual_predicted_denominator',et,dp),('predicted_residual_oracle_denominator',ep,dt)]:
  result=rollout(p,Objective(ee,dd,mass,1.,precompute=False),f['risk_scores'],pool,compare=False);interventions[label]=dict(nmse=result['trace'][selected['steps']]['nmse'],trace=result['trace'])
 result=dict(event_id=eid,split=split(eid),projection_top_actions=stats,risk_set_residual_cosine_mean=float(cos.mean()),risk_set_residual_cosine_median=float(np.median(cos)),denominator_relative_mae=float(np.mean(np.abs(zp-zt)/zt)),risk_nmse=p.value(x),oracle_diagnostic_interventions=interventions)
 dump(OUT/'diagnostics'/f'{eid}.json',result);return result
if __name__=='__main__':
 ids=sorted(p.name for p in (ROOT/'contributions').iterdir())
 with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:rows=list(ex.map(runone,ids))
 summary={}
 for sp in ('train','validation','evaluation'):
  rr=[r for r in rows if r['split']==sp];summary[sp]=dict(n=len(rr),median_set_residual_cosine=float(np.median([r['risk_set_residual_cosine_median'] for r in rr])),mean_denominator_relative_mae=float(np.mean([r['denominator_relative_mae'] for r in rr])),oracle_interventions={label:float(np.mean([100*(1-r['oracle_diagnostic_interventions'][label]['nmse']/r['risk_nmse']) for r in rr])) for label in rr[0]['oracle_diagnostic_interventions']})
 dump(OUT/'diagnostics_summary.json',summary);print(json.dumps(summary,indent=2))
