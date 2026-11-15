from scipy.stats import spearmanr
from .core import *
from .fit import ids,deltas,summary_metric,MODES

def run():
 lock=json.loads((OUT/'model_lock.json').read_text());models={m['name']:np.load(OUT/'models'/(m['name']+'.npz'))['coef'] for m in lock['models']};models['pair_only']=np.zeros(85);static=np.load(OUT/'models/static_pair.npz')['coef'];reg=[];ranking=[];selection=[]
 for sp in ('train','validation','evaluation'):
  for eid in ids(sp):
   d=np.load(OUT/'events'/eid/'probes.npz');phi=d['phi'];G=d['gains'];J=d['nmse'];sigma=float(d['scale']);S,A=G.shape;pair=np.column_stack((d['pair'],np.ones(A)));online_base=pair@static;true_base=G[0]/sigma;delta_phi=phi-phi[:1];oracle=np.maximum(G.max(1),0);out={}
   for mode in MODES:
    X,y=deltas(d,mode)
    for name,coef in models.items():
     if name!='pair_only' and not name.startswith(mode+'__'):continue
     pred=X@coef;corr=spearmanr(pred,y).statistic if np.ptp(pred)>1e-15 and np.ptp(y)>1e-15 else np.nan
     reg.append(dict(event_id=eid,split=sp,model=name,mode=mode,mse=float(np.mean((pred-y)**2)),target_energy=float(np.mean(y**2)),target_mean=float(y.mean()),correlation=float(corr) if np.isfinite(corr) else None))
   for name,coef in models.items():
    change=delta_phi@coef;out[name]=change
    for panel,base in [('online_linear',online_base),('oracle_initial_anchor',true_base)]:
     scores=base[None]+change;chosen=scores.argmax(1);g=G[np.arange(S),chosen];baseid=int(base.argmax());baseg=G[:,baseid];pos=np.maximum(g,0);bpos=np.maximum(baseg,0);rho=[]
     for s in range(S):
      c=spearmanr(scores[s],G[s]).statistic
      if np.isfinite(c):rho.append(float(c))
      selection.append(dict(event_id=eid,split=sp,model=name,panel=panel,state=s,action=int(chosen[s]),gain=float(g[s]),baseline_action=baseid,baseline_gain=float(baseg[s]),oracle_gain=float(oracle[s]),state_nmse=float(J[s])))
     ranking.append(dict(event_id=eid,split=sp,model=name,panel=panel,states=S,forced_mean_gain=float(g.mean()),forced_state_relative_pct=float(np.mean(g/J)*100),positive_filtered_gain=float(pos.mean()),positive_filtered_state_relative_pct=float(np.mean(pos/J)*100),baseline_positive_filtered_gain=float(bpos.mean()),oracle_gain=float(oracle.mean()),correction_gain=float((pos-bpos).mean()),correction_state_relative_pp=float(np.mean((pos-bpos)/J)*100),regret=float((oracle-pos).mean()),rank_spearman=float(np.mean(rho)) if rho else None,degraded_states=int((g< -1e-12).sum()),positive_states=int((g>1e-12).sum())))
   dest=OUT/'predictions';dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'{eid}.npz',**out)
 summary=[]
 for sp in ('train','validation','evaluation'):
  for mode in MODES:
   for name in ['pair_only']+[m['name'] for m in lock['models'] if m['mode']==mode]:
    rows=[r for r in reg if r['split']==sp and r['model']==name and r['mode']==mode];summary.append(dict(split=sp,model=name,mode=mode,**summary_metric(rows)))
 dump(OUT/'regression_per_event.json',reg);dump(OUT/'regression_summary.json',summary);dump(OUT/'ranking_per_event.json',ranking);dump(OUT/'selected_actions.json',selection);print(json.dumps([r for r in summary if r['split']=='evaluation'],indent=2))
if __name__=='__main__':run()
