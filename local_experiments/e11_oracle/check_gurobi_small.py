"""Independent small MIQP cross-check within the existing restricted license."""
import json,numpy as np,gurobipy as gp
from pathlib import Path
from .solver import setup,objective,exhaustive
ROOT=Path('/tmp/e11_results');rows=[];rng=np.random.default_rng(17)
full=dict(np.load(ROOT/'contributions/horse_field_s101_c2_l1_s0.npz'))
for kind in ('synthetic','real_reduced'):
 for P,K in ((12,2),(16,3),(24,4)):
  Q,d=4,8
  if kind=='synthetic':
   Z=np.exp(rng.normal(scale=.3,size=(Q,P)));Zh=Z*np.exp(rng.normal(scale=.2,size=(Q,P)));N=Z[:,:,None]*rng.normal(size=(Q,P,d));Nh=N+rng.normal(scale=.2,size=N.shape);data=dict(N=N,Z=Z,Nh=Nh,Zh=Zh)
  else:
   pages=np.r_[int(full['own']),np.delete(np.arange(full['Z'].shape[1]),int(full['own']))[:P-1]];data={n:full[n][:Q,pages,:d] if n in ('N','Nh') else full[n][:Q,pages] for n in ('N','Z','Nh','Zh')}
  val,exact,count=exhaustive(data,K,0);free,a,b,dn,dz,lo,hi,od,cd=setup(data,K,0);F=P-1;m=gp.Model();m.Params.OutputFlag=0;m.Params.NonConvex=0;m.Params.MIPGap=5e-7;m.Params.MIPGapAbs=5e-11;m.Params.FeasibilityTol=1e-8;m.Params.IntFeasTol=1e-8;m.Params.Threads=8;m.Params.Seed=17;m.Params.TimeLimit=120
  x=m.addVars(F,vtype=gp.GRB.BINARY);y=m.addVars(Q,lb={q:lo[q] for q in range(Q)},ub={q:hi[q] for q in range(Q)});w=m.addVars(Q,F,lb=0.);t=m.addVars(Q,d,lb=-gp.GRB.INFINITY);m.addConstr(gp.quicksum(x.values())==K-1)
  for q in range(Q):
   for p in range(F):
    m.addConstr(w[q,p]>=lo[q]*x[p]);m.addConstr(w[q,p]<=hi[q]*x[p]);m.addConstr(w[q,p]>=y[q]-hi[q]*(1-x[p]));m.addConstr(w[q,p]<=y[q]-lo[q]*(1-x[p]))
   m.addConstr(b[q]*y[q]+gp.quicksum(dz[q,p]*w[q,p] for p in range(F))==1)
   for j in range(d):m.addConstr(t[q,j]==a[q,j]*y[q]+gp.quicksum(dn[q,p,j]*w[q,p] for p in range(F))-od[q,j])
  m.setObjective(gp.quicksum(v*v for v in t.values())/cd);m.optimize();mask=np.zeros(P,bool);mask[0]=True;mask[free]=[x[p].X>.5 for p in range(F)];ub=objective(data,mask)
  passed=abs(ub-val)<=max(1e-10,1e-6*abs(val)) and m.ObjBound<=val+1e-10
  row=dict(scope=kind,P=P,K=K,Q=Q,d=d,enumeration_status='EXHAUSTIVE_CHECKED',enumerated_sets=count,exhaustive_value=val,UB=ub,LB=m.ObjBound,solver_status=m.Status,solver_version=gp.gurobi.version(),NonConvex=0,passed=passed,mask=np.where(mask)[0].tolist());rows.append(row);(ROOT/'gurobi_small_checks.json').write_text(json.dumps(rows,indent=2));print(kind,P,passed,flush=True)
assert all(x['passed'] for x in rows)
