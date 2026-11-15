import json,time,csv,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from ..e12_swap_prediction.core import DATA,split,objective
from ..set_oracle_v1.common import dump,sha
from ..set_oracle_v1.problem import ResidualProblem
from ..set_oracle_v1.search import exchange
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/e16_true_mass_v1'
def select(scores,K,own):
 order=np.argsort(-scores,kind='stable');x=np.zeros(len(scores),bool);x[own]=True;x[order[order!=own][:K-1]]=True;return x

def one(eid):
 out=OUT/'events'/eid;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 start=time.monotonic();src=DATA/'contributions'/eid;p=ResidualProblem(src);f=dict(np.load(src/'summary_and_route_features.npz'));raw=dict(np.load(src/'residuals_fp64.npz'));oldpath=DATA/'searches/v1'/eid/'result.json';old=json.loads(oldpath.read_text());qi=f['probe_indices'].astype(int);exact=raw['exact_mass'];approx=raw['summary_mass']/raw['summary_mass'].sum(1,keepdims=True)
 assert np.max(abs(exact.sum(1)-1))<1e-12
 # Independent normalized logZ reconstruction verifies true page mass identity.
 z=f['exact_page_logz'].astype(float);z=np.exp(z-z.max(1,keepdims=True));z/=z.sum(1,keepdims=True);mass_error=float(np.max(abs(z-exact)));assert mass_error<1e-10
 scores={'true_mass16':exact[qi].mean(0),'true_mass64':exact.mean(0),'summary_mass16_fp64':approx[qi].mean(0),'summary_mass64_fp64':approx.mean(0)}
 masks={n:select(s,p.K,p.own) for n,s in scores.items()};masks.update(summary_mass16=raw['mass'].copy(),risk16=raw['risk'].copy())
 for source in ['greedy','risk_exchange','mass_exchange','greedy_exchange']:
  x=np.zeros(p.P,bool);x[old['masks'][source]]=True;masks[source]=x
 traces={};certs={}
 for name in ['true_mass16','true_mass64']:
  x,trace,states,cert,_,_=exchange(p,masks[name],False);masks[name+'_exchange']=x;traces[name]=trace;certs[name]=cert
 values={};rows=[]
 for name,x in masks.items():
  assert x.sum()==p.K and x[p.own];j=objective(raw,x);assert abs(j-p.value(x))<1e-12;values[name]=j
  original={'summary_mass16':'mass','risk16':'risk'}.get(name,name)
  if original in old['values']:assert abs(j-old['values'][original])<max(1e-11,abs(j)*1e-8)
  rows.append(dict(event_id=eid,split=split(eid),layer=old['identity']['layer'],chunk=old['identity']['chunk'],step=old['identity']['step'],method=name,nmse=j,count=int(x.sum()),true_mass64_captured=float(exact[:,x].sum(1).mean()),true_mass16_captured=float(exact[qi][:,x].sum(1).mean()),changed_from_risk=int(np.sum(x&~masks['risk16'])),changed_from_summary_mass=int(np.sum(x&~masks['summary_mass16']))))
 # True mass top-k is optimal only for its mass objective, not NMSE.
 for name,qs in [('true_mass16',qi),('true_mass64',np.arange(64))]:
  mass=exact[qs].mean(0);v=mass[masks[name]].sum()
  assert all(v>=mass[x].sum()-1e-12 for x in masks.values())
 np.savez_compressed(out/'masks_scores.npz',**{f'mask_{k}':v for k,v in masks.items()},**{f'score_{k}':v for k,v in scores.items()})
 result=dict(event_id=eid,split=split(eid),values=values,rows=rows,trajectories=traces,local_certificates=certs,true_vs_summary_rank_spearman16=float(spearmanr(scores['true_mass16'],scores['summary_mass16_fp64']).statistic),true_mass16_64_page_difference=int(np.sum(masks['true_mass16']&~masks['true_mass64'])),fp64_summary_deployed_page_difference=int(np.sum(masks['summary_mass16']&~masks['summary_mass16_fp64'])),same_exchange_masks=bool(np.array_equal(masks['true_mass16_exchange'],masks['risk_exchange'])),mass_logz_max_error=mass_error,wall_seconds=time.monotonic()-start,source_hashes={str(path.relative_to(ROOT)):sha(path) for path in [src/'residuals_fp64.npz',src/'summary_and_route_features.npz',oldpath]})
 dump(out/'result.json',result);return result

def table(name,rows):
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def summarize(results):
 rows=[x for r in results for x in r['rows']];table('per_instance.csv',rows);summ=[];pairs=[]
 for sp in ['train','validation','evaluation','all']:
  rr=[r for r in results if sp=='all' or r['split']==sp]
  for method in rr[0]['values']:
   v=np.array([r['values'][method] for r in rr]);risk=np.array([r['values']['risk16'] for r in rr]);summ.append(dict(split=sp,method=method,n=len(rr),mean_nmse=float(v.mean()),median_nmse=float(np.median(v)),mean_relative_improvement_vs_risk_pct=float(np.mean(1-v/risk)*100),median_relative_improvement_vs_risk_pct=float(np.median(1-v/risk)*100),pooled_improvement_vs_risk_pct=float((1-v.sum()/risk.sum())*100),better_than_risk=int(np.sum(v<risk-1e-12)),worse_than_risk=int(np.sum(v>risk+1e-12))))
  for base,method in [('summary_mass16','true_mass16'),('summary_mass16_fp64','true_mass16'),('summary_mass64_fp64','true_mass64'),('risk16','true_mass16'),('true_mass16','true_mass64'),('true_mass16','risk_exchange'),('true_mass64','risk_exchange'),('true_mass16','true_mass16_exchange'),('true_mass64','true_mass64_exchange')]:
   b=np.array([r['values'][base] for r in rr]);v=np.array([r['values'][method] for r in rr]);gain=1-v/b;pairs.append(dict(split=sp,baseline=base,method=method,n=len(rr),mean_relative_improvement_pct=float(gain.mean()*100),median_relative_improvement_pct=float(np.median(gain)*100),p10_relative_improvement_pct=float(np.quantile(gain,.1)*100),pooled_improvement_pct=float((1-v.sum()/b.sum())*100),better=int(np.sum(v<b-1e-12)),worse=int(np.sum(v>b+1e-12))))
 table('summary.csv',summ);table('paired_comparisons.csv',pairs)
 dump(OUT/'summary.json',dict(methods=summ,pairs=pairs,events=len(results),max_mass_logz_error=max(r['mass_logz_max_error'] for r in results),max_local_direct_error=max(c['max_gram_direct_error'] for r in results for c in r['local_certificates'].values()),same_true16_and_risk_exchange_masks=sum(r['same_exchange_masks'] for r in results),mean_true_summary_spearman16=float(np.mean([r['true_vs_summary_rank_spearman16'] for r in results])),max_fp64_summary_deployed_page_difference=max(r['fp64_summary_deployed_page_difference'] for r in results)))
 print(json.dumps([r for r in pairs if r['split']=='evaluation'],indent=2),flush=True)
def run():
 OUT.mkdir(parents=True,exist_ok=True)
 if not (OUT/'protocol.json').exists():dump(OUT/'protocol.json',dict(created=time.time(),scope='E16 extra true mass comparison; frozen144 headtiles, original baseline states only, no CF states/new QKV/video/training/global certificates',mass='exact page Z / sum_all_page_exact_Z per query, then mean over probes; all legal pages, own protected, fixed10%',main='true_mass16 vs deployed summary_mass16 and risk16; true_mass64 separately information-control, summary_mass64_fp64 matching control',precision='BF16 deployed input contributions in FP64; additional summary_mass16_fp64 checks numerical alignment; original online masks preserved',objective='all64Query true mixed output NMSE, varying denominator; true mass is offline KV-reading oracle not online router',search='reuse original greedy and mass/risk/greedy Exchange; additionally start complete one-swap descent from true_mass16 and true_mass64, all unselected pages; certified local only, never global optimum',split='city101 train36 city303 validation36 horse101/303 evaluation72; no fit/tuning; report all and split',dataset_manifest_sha=sha(DATA/'manifest.json')))
 ids=sorted(p.name for p in (DATA/'contributions').iterdir() if (p/'residuals_fp64.npz').exists())
 with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
  results=[]
  for i,r in enumerate(ex.map(one,ids),1):results.append(r);print(i,r['event_id'],flush=True)
 summarize(results)
if __name__=='__main__':run()
