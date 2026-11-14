import argparse,time,json,hashlib
from pathlib import Path
import numpy as np
from .common import *
from .problem import ResidualProblem

def greedy(p):
 x=np.zeros(p.P,bool);x[p.own]=True;trace=[dict(step=0,count=1,nmse=p.value(x),added=None,delta=None)]
 for step in range(1,p.K):
  a=np.where(~x)[0];E,z,j=p.state(x);v=E[:,None,:]-p.e[:,a];zz=z[:,None]+p.d[:,a];scores=np.sum(np.square(v).sum(-1)/zz**2,axis=0)/p.C;chosen=int(a[np.argmin(scores)]);x[chosen]=True;after=p.value(x);assert abs(after-float(scores.min()))<max(1e-12,abs(after)*1e-9);trace.append(dict(step=step,count=int(x.sum()),nmse=after,added=chosen,delta=j-after,all_increase=bool(scores.min()>j)))
 return x,trace

def exchange(p,start,keep_surfaces):
 x=start.copy();j=p.value(x);trace=[dict(step=0,nmse=j,remove=None,add=None,delta=0.)];states=[x.copy()];surfaces=[];top=[];rs=[];adds=[];tol=None
 for step in range(1000):
  j=p.value(x);tol=max(1e-12,1e-9*abs(j));r,a,scores=p.swaps(x);gain=j-scores
  if keep_surfaces:surfaces.append(gain.copy());rs.append(r.copy());adds.append(a.copy())
  order=np.argsort(-gain.flatten(),kind='stable')[:20];best=int(order[0]);ri,ai=np.unravel_index(best,gain.shape);top.append(dict(step=step,state_sha256=hashlib.sha256(x.tobytes()).hexdigest(),best_gain=float(gain[ri,ai]),first_second_gap=float(gain.flatten()[order[0]]-gain.flatten()[order[1]]),top20=[dict(remove=int(r[k//len(a)]),add=int(a[k%len(a)]),gain=float(gain.flatten()[k])) for k in order]))
  if scores[ri,ai]>=j-tol:
   _,_,independent=p.swaps(x,direct=True);difference=float(np.max(np.abs(independent-scores)));assert difference<max(1e-11,abs(j)*1e-8);ri,ai=np.unravel_index(np.argmin(independent),independent.shape)
   if independent[ri,ai]>=j-tol:cert=dict(status='ONE_SWAP_LOCAL_TOL',neighbors=len(r)*len(a),objective=j,best_neighbor=float(independent.min()),tolerance=tol,max_gram_direct_error=difference);break
  remove=int(r[ri]);add=int(a[ai]);x[remove]=False;x[add]=True;after=p.value(x);assert after<j-tol;trace.append(dict(step=step+1,nmse=after,remove=remove,add=add,delta=j-after));states.append(x.copy())
 else:raise RuntimeError('local search step limit')
 return x,trace,states,cert,dict(gains=np.stack(surfaces),removed=np.stack(rs),added=np.stack(adds),state_masks=np.stack(states)) if keep_surfaces else None,top

def d02(p,x,history,ages):
 E,z,j=p.state(x);tail=~x;energy=np.square(p.e).sum(-1);A=energy[:,tail].sum(1);norm=np.square(E).sum(1);EH=p.e[:,tail&history].sum(1);EC=p.e[:,tail&~history].sum(1);AH=energy[:,tail&history].sum(1);AC=energy[:,tail&~history].sum(1);HH=np.square(EH).sum(1)-AH;CC=np.square(EC).sum(1)-AC;HC=2*np.sum(EH*EC,axis=1);weighted=np.stack((AH,AC,HH,CC,HC),-1)/z[:,None]**2/p.C
 assert abs(float(weighted.sum())-j)<1e-10
 ageE=np.stack([p.e[:,tail&(ages==age)].sum(1) for age in range(6)]);ageA=np.stack([energy[:,tail&(ages==age)].sum(1) for age in range(6)]);age_inner=np.einsum('aqd,bqd->qab',ageE,ageE)
 return dict(A=A,C=norm-A,Z=z,kappa=norm/(A+1e-12),weighted_components=weighted,age_A=ageA,age_inner_products=age_inner,nmse=np.array(j),fixed_dense_denominator_nmse=np.array(norm.sum()/p.C))

def run(eid):
 root=ROOT/'contributions'/eid;out=ROOT/'searches'/'v1'/eid;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return eid
 start=time.monotonic();cpu=time.process_time();p=ResidualProblem(root);identity=json.loads((root/'identity.json').read_text());online=dict(np.load(root/'summary_and_route_features.npz'));methods={n:p.data[n].copy() for n in ('mass','risk')};g,gt=greedy(p);methods['greedy']=g;trajectories={};certs={};states={};alltop={}
 for name in ('risk','mass','greedy'):
  x,trace,ss,cert,surface,top=exchange(p,methods[name],name=='risk');methods[name+'_exchange']=x;trajectories[name]=trace;certs[name]=cert;states[name]=ss;alltop[name]=top
  if surface is not None:np.savez_compressed(out/'candidate_gain_surfaces.npz',**surface)
 checks=[];direct_data=dict(np.load(ROOT/'_validation_spool'/f'{eid}.npz'));validation_masks=list(methods.items())+[(f'risk_step{i}',x) for i,x in enumerate(states['risk'])];rng=np.random.default_rng(17)
 for name,x in validation_masks:
  true=direct(direct_data,x);res=p.value(x);err=abs(true-res);assert err<max(1e-11,abs(true)*1e-8);checks.append(dict(name=name,direct_nmse=true,residual_nmse=res,absolute_error=err))
 for qsel in (np.array([0]),np.array([3,17,31,63]),np.sort(rng.choice(64,16,replace=False))):
  x=methods['risk_exchange'];E,z,_=p.state(x);pred=float(np.sum(np.square(E[qsel]).sum(1)/z[qsel]**2)/(np.square(p.od[qsel]).sum()+1e-12));true=direct(direct_data,x,qsel);assert abs(true-pred)<max(1e-11,abs(true)*1e-8);checks.append(dict(name='query_subset',queries=qsel.tolist(),direct_nmse=true,residual_nmse=pred,absolute_error=abs(true-pred)))
 names=[];decomps=[]
 for name,x in methods.items():names.append(name);decomps.append(d02(p,x,online['page_is_history'],online['page_age']))
 for name,ss in states.items():
  for i,x in enumerate(ss):names.append(f'{name}_exchange_step{i}');decomps.append(d02(p,x,online['page_is_history'],online['page_age']))
 np.savez_compressed(out/'D02.npz',names=np.array(names),**{n:np.stack([d[n] for d in decomps]) for n in decomps[0]})
 values={n:p.value(x) for n,x in methods.items()};best=min(values,key=values.get);gain=values['risk']-values['risk_exchange'];rt=trajectories['risk'];early={str(k):(values['risk']-rt[min(k,len(rt)-1)]['nmse'])/gain if gain>1e-12 else None for k in (1,2,4)}
 result=dict(event_id=eid,identity=identity,P=p.P,K=p.K,Q=64,own=p.own,values=values,masks={n:np.where(x)[0].tolist() for n,x in methods.items()},trajectories=trajectories,greedy_trace=gt,local_certificates=certs,top_candidates=alltop,best_objective=values[best],best_method=best,best_mask=np.where(methods[best])[0].tolist(),early_risk_gain_fraction=early,greedy_negative_steps=sum(t['delta'] is not None and t['delta']<0 for t in gt),initialization_same_mask=bool(np.array_equal(methods['risk_exchange'],methods['mass_exchange']) and np.array_equal(methods['risk_exchange'],methods['greedy_exchange'])),local_worse_than_other_feasible={n:values[n+'_exchange']-min(values.values()) for n in ('mass','risk','greedy')},wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,contribution_files=identity['files'])
 dump(out/'validation.json',dict(passed=True,checks=checks,max_error=max(t['absolute_error'] for t in checks)));dump(out/'result.json',result);(ROOT/'_validation_spool'/f'{eid}.npz').unlink();print('SEARCH_DONE',eid,values['risk'],values['risk_exchange'],flush=True);return eid
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--event',required=True);run(a.parse_args().event)
