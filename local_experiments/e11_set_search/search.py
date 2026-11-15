import argparse,json,time,hashlib,csv
from pathlib import Path
import numpy as np
from ..e11_compact.core import Problem
ROOT=Path('/tmp/e11_set_search')

def greedy(p):
 x=np.zeros(p.P,bool);x[p.own]=True;trace=[dict(step=0,count=1,nmse=p.value(x),added=None,delta=None,candidates=0)];checks=[];start=time.monotonic()
 for step in range(1,p.K):
  free=np.where(~x)[0];E,z,j=p.state(x);ee=E[:,None,:]-p.e[:,free];zz=z[:,None]+p.d[:,free];assert np.all(zz>0)
  score=np.sum(np.square(ee).sum(-1)/zz**2,axis=0)/p.C
  # Independent original mixed N/Z evaluation of EVERY addition, not just the winner.
  N,Z,Nh,Zh=(p.data[k] for k in ('N','Z','Nh','Zh'));baseN=np.where(x[None,:,None],N,Nh).sum(1);baseZ=np.where(x[None],Z,Zh).sum(1);out=(baseN[:,None,:]+N[:,free]-Nh[:,free])/(baseZ[:,None]+Z[:,free]-Zh[:,free])[:,:,None];direct=np.square(out-p.od[:,None]).sum((0,2))/p.C
  err=float(np.max(np.abs(score-direct)));assert err<max(1e-12,1e-9*float(np.max(direct)));checks.append(err)
  winner=int(free[np.argmin(score)]);before=p.value(x);x[winner]=True;after=p.value(x);assert abs(after-float(direct[np.argmin(score)]))<1e-12
  trace.append(dict(step=step,count=int(x.sum()),nmse=after,added=winner,delta=before-after,candidates=len(free),all_additions_increase=bool(np.min(direct)>before),wall_seconds=time.monotonic()-start))
 return x,dict(method='greedy',trace=trace,mask=np.where(x)[0].tolist(),final=p.value(x),wall_seconds=time.monotonic()-start,max_all_candidate_validation_error=max(checks),negative_gain_steps=sum(r['delta'] is not None and r['delta']<0 for r in trace),candidate_evaluations=sum(r['candidates'] for r in trace))

def replay(p,start,run):
 x=start.copy();j=p.value(x);rows=[dict(step=0,nmse=j,delta=0.,relative_to_initial=1.,remove=None,add=None)];initial=j
 for i,t in enumerate(run['trace'],1):
  assert x[t['remove']] and not x[t['add']] and t['remove']!=p.own;x[t['remove']]=False;x[t['add']]=True;v=p.value(x);assert abs(v-t['after'])<1e-12 and v<j
  rows.append(dict(step=i,nmse=v,delta=j-v,relative_to_initial=v/initial,remove=t['remove'],add=t['add']));j=v
 assert set(np.where(x)[0])==set(run['mask']);assert abs(j-run['final'])<1e-12
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['horse_field','traffic_crossing'],required=True);ap.add_argument('--layer',type=int,required=True);a=ap.parse_args();r=ROOT/a.case/f'l{a.layer}';r.mkdir(parents=True,exist_ok=True);path=Path(f'/tmp/e11_results/contributions/{a.case}_s101_c5_l{a.layer}_s0.npz');data=dict(np.load(path));begin=time.monotonic();p=Problem(data);runs={};curves={};masks={k:data[k].copy() for k in ('mass','risk')}
 if a.case=='horse_field':
  oldroot=Path(f'/tmp/e11_compact/l{a.layer}');old=json.loads((oldroot/'local_result.json').read_text());identity=json.loads((oldroot/'validation.json').read_text());assert identity['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
  for name in ('mass','risk'):
   rr=next(z for z in old['runs'] if z['label']==name);runs[name+'_exchange']={**rr,'reused':True};xx=np.zeros(410,bool);xx[rr['mask']]=True;masks[name+'_exchange']=xx;curves[name+'_exchange']=replay(p,data[name],rr)
  xx=np.zeros(410,bool);xx[old['best_mask']]=True;masks['legacy_multistart_best']=xx
 else:
  for name in ('mass','risk'):
   xx,rr=p.descend(data[name],name);runs[name+'_exchange']={**rr,'reused':False};masks[name+'_exchange']=xx;curves[name+'_exchange']=replay(p,data[name],rr)
 x,g=greedy(p);masks['greedy']=x;xx,rr=p.descend(x,'greedy');masks['greedy_exchange']=xx;runs['greedy_exchange']={**rr,'reused':False};curves['greedy_exchange']=replay(p,x,rr)
 values={k:p.value(v) for k,v in masks.items()};bestname=min(values,key=values.get);best=masks[bestname]
 out=dict(case=a.case,layer=a.layer,P=410,Q=64,K=41,own=p.own,head=int(data['head']),input_path=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),values=values,masks={k:np.where(v)[0].tolist() for k,v in masks.items()},greedy=g,exchange_runs=runs,exchange_curves=curves,best_method=bestname,best_objective=values[bestname],best_mask=np.where(best)[0].tolist(),baseline_mass=values['mass'],baseline_risk=values['risk'],wall_seconds=time.monotonic()-begin)
 for v in masks.values():assert v.sum()==41 and v[p.own]
 (r/'search_result.json').write_text(json.dumps(out,indent=2));(r/'local_result.json').write_text(json.dumps(out,indent=2))
 with (r/'exchange_steps.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['method','step','nmse','delta','relative_to_initial','remove','add']);w.writeheader()
  for method,rows in curves.items():w.writerows(dict(method=method,**row) for row in rows)
 with (r/'greedy_steps.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['step','count','nmse','added','delta','candidates','all_additions_increase','wall_seconds']);w.writeheader();w.writerows(g['trace'])
 print(json.dumps(dict(case=a.case,layer=a.layer,values=values,negative_greedy_steps=g['negative_gain_steps'],seconds=out['wall_seconds'])),flush=True)
if __name__=='__main__':main()
