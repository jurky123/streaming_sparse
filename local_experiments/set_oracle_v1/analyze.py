import json,csv,time
from pathlib import Path
import numpy as np
from .common import *
def stats(v):
 a=np.asarray([x for x in v if x is not None],float)
 if not len(a):return dict(n=0)
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),max=float(a.max()))
def table(path,rows):
 if not rows:return
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 manifest=json.loads((ROOT/'manifest.json').read_text());assert manifest['state']=='FROZEN_VALIDATED';rows=[];dec=[];actions=[];curves=[];masks=[]
 for ev in manifest['events']:
  eid=ev['event_id'];r=ROOT/'searches'/'v1'/eid;s=json.loads((r/'result.json').read_text());v=s['values'];base=min(v['mass'],v['risk']);trace=s['trajectories']['risk'];row={k:ev[k] for k in ('event_id','case','seed','chunk','layer','step','head','tile','P','K')};row.update(v);row.update(gain_vs_mass=1-v['risk_exchange']/v['mass'],gain_vs_risk=1-v['risk_exchange']/v['risk'],gain_vs_stronger=1-v['risk_exchange']/base,risk_swaps=len(trace)-1,mass_swaps=len(s['trajectories']['mass'])-1,greedy_swaps=len(s['trajectories']['greedy'])-1,same_final_mask=s['initialization_same_mask'],greedy_increasing_steps=s['greedy_negative_steps'],first1=s['early_risk_gain_fraction']['1'],first2=s['early_risk_gain_fraction']['2'],first4=s['early_risk_gain_fraction']['4'],risk_local_minus_best_feasible=s['local_worse_than_other_feasible']['risk'],search_wall_seconds=s['wall_seconds'],search_cpu_seconds=s['cpu_seconds']);rows.append(row)
  for name,tr in s['trajectories'].items():
   for t in tr:curves.append(dict(event_id=eid,initialization=name,**t,relative_to_initial=t['nmse']/tr[0]['nmse']))
  for method,ids in s['masks'].items():masks.append(dict(event_id=eid,method=method,pages=ids,nmse=v[method]))
  d=dict(np.load(r/'D02.npz'));names=d['names'].tolist();ir=names.index('risk');ix=names.index('risk_exchange');wr=d['weighted_components'][ir];wx=d['weighted_components'][ix];riskA=float(wr[:,:2].sum());newA=float(wx[:,:2].sum());riskC=float(wr[:,2:].sum());newC=float(wx[:,2:].sum());counter=float(np.sum(wx.sum(-1)*(d['Z'][ix]/d['Z'][ir])**2));dd=dict(event_id=eid,case=ev['case'],layer=ev['layer'],chunk=ev['chunk'],step=ev['step'],risk_nmse=v['risk'],exchange_nmse=v['risk_exchange'],risk_A_weighted=riskA,exchange_A_weighted=newA,risk_C_weighted=riskC,exchange_C_weighted=newC,delta_A=newA-riskA,delta_C=newC-riskC,risk_Z_ratio_mean=float(d['Z'][ir].mean()),exchange_Z_ratio_mean=float(d['Z'][ix].mean()),exchange_residual_risk_denominator_nmse=counter,denominator_effect=v['risk_exchange']-counter,net_negative_cross_fraction_risk=float(np.mean(d['C'][ir]<0)),net_negative_cross_fraction_exchange=float(np.mean(d['C'][ix]<0)))
  with np.load(ROOT/'contributions'/eid/'residuals_fp64.npz') as rr:energy=float(rr['C'])
  deltaA=float(np.sum((d['A'][ix]-d['A'][ir])/d['Z'][ir]**2)/energy);deltaC=float(np.sum((d['C'][ix]-d['C'][ir])/d['Z'][ir]**2)/energy);assert abs(deltaA+deltaC+dd['denominator_effect']-(v['risk_exchange']-v['risk']))<1e-10
  dd.update(delta_A_at_risk_denominator=deltaA,delta_C_at_risk_denominator=deltaC,risk_A_dense_normalized=float(d['A'][ir].sum()/energy),exchange_A_dense_normalized=float(d['A'][ix].sum()/energy),risk_C_dense_normalized=float(d['C'][ir].sum()/energy),exchange_C_dense_normalized=float(d['C'][ix].sum()/energy))
  parts={'independent_energy':-deltaA,'cross_page_interaction':-deltaC,'denominator':-dd['denominator_effect']};dd['largest_positive_component']=max(parts,key=parts.get) if v['risk']-v['risk_exchange']>1e-12 else 'no_gain'
  for col,n in enumerate(('A_history','A_current','C_HH','C_CC','C_HC')):dd['risk_'+n]=float(wr[:,col].sum());dd['exchange_'+n]=float(wx[:,col].sum())
  dec.append(dd);online=dict(np.load(ROOT/'contributions'/eid/'summary_and_route_features.npz'));rank=np.empty(ev['P'],int);rank[online['risk_rank']]=np.arange(ev['P']);x=np.zeros(ev['P'],bool);x[s['masks']['risk']]=True
  for i,top in enumerate(s['top_candidates']['risk']):
   best=top['top20'][0];remove,add=best['remove'],best['add'];unselected_order=[int(p) for p in online['risk_rank'] if not x[p]];selected_order=[int(p) for p in online['risk_rank'] if x[p] and p!=s['own']];actions.append(dict(event_id=eid,iteration=i,state_sha256=top['state_sha256'],current_nmse=trace[i]['nmse'],best_gain=top['best_gain'],first_second_gap=top['first_second_gap'],positive=top['best_gain']>max(1e-12,trace[i]['nmse']*1e-9),remove=remove,add=add,remove_risk_rank=int(rank[remove]),add_risk_rank=int(rank[add]),remove_rank_among_selected=selected_order.index(remove),add_rank_among_unselected=unselected_order.index(add),remove_history=bool(online['page_is_history'][remove]),add_history=bool(online['page_is_history'][add]),remove_age=int(online['page_age'][remove]),add_age=int(online['page_age'][add])))
   if i+1<len(trace):x[trace[i+1]['remove']]=False;x[trace[i+1]['add']]=True
 analysis=ROOT/'analyses';table(analysis/'per_instance.csv',rows);table(analysis/'exchange_steps.csv',curves);table(analysis/'cancellation/per_instance.csv',dec);table(analysis/'actions/best_per_iteration.csv',actions);dump(analysis/'per_instance.json',rows);dump(analysis/'final_masks.json',masks)
 groups=[]
 for field in ('case','seed','chunk','layer','step','head','P'):
  for value in sorted({r[field] for r in rows}):
   subset=[r for r in rows if r[field]==value];g=stats([r['gain_vs_stronger'] for r in subset]);groups.append(dict(dimension=field,value=value,**g,positive=sum(r['gain_vs_stronger']>1e-8 for r in subset),same_final_mask=sum(r['same_final_mask'] for r in subset)))
 table(analysis/'stratified.csv',groups)
 summary=dict(events=len(rows),independent_base_trajectories=4,gain_vs_stronger=stats([r['gain_vs_stronger'] for r in rows]),gain_vs_risk=stats([r['gain_vs_risk'] for r in rows]),positive_vs_stronger=sum(r['gain_vs_stronger']>1e-8 for r in rows),negative_vs_stronger=sum(r['gain_vs_stronger']<-1e-8 for r in rows),risk_swap_count=stats([r['risk_swaps'] for r in rows]),same_final_masks=sum(r['same_final_mask'] for r in rows),risk_local_worse_than_other_feasible=sum(r['risk_local_minus_best_feasible']>max(1e-12,1e-9*r['risk_exchange']) for r in rows),early_gain={k:stats([r['first'+k] for r in rows]) for k in ('1','2','4')},greedy_nonmonotone_events=sum(r['greedy_increasing_steps']>0 for r in rows),A_increases=sum(d['delta_A']>1e-12 for d in dec),C_decreases=sum(d['delta_C']<-1e-12 for d in dec),A_increases_C_decreases=sum(d['delta_A']>1e-12 and d['delta_C']<-1e-12 for d in dec),denominator_adverse=sum(d['denominator_effect']>1e-12 for d in dec),denominator_beneficial=sum(d['denominator_effect']<-1e-12 for d in dec),total_search_cpu_seconds=sum(r['search_cpu_seconds'] for r in rows),global_status='separate; representative bounds may still be running',scope='144 correlated head-tiles from4 trajectories; two development-seen prompts; no unseen-test claim')
 summary['dominant_gain_component']={k:sum(d['largest_positive_component']==k for d in dec) for k in ('independent_energy','cross_page_interaction','denominator','no_gain')};summary['C_decreases_at_fixed_risk_denominator']=sum(d['delta_C_at_risk_denominator']<-1e-12 for d in dec);dump(analysis/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
