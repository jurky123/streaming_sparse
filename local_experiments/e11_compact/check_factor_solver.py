"""Exhaustive audit of the factorized convex lower model, not original OPT."""
import itertools,json
from pathlib import Path
import numpy as np
from pyscipopt import Model,quicksum
from .core import Problem
from .relaxation import build_h
rng=np.random.default_rng(17);rows=[]
for P,K,Q,D in [(12,2,4,8),(16,3,64,128),(24,4,64,128)]:
 Z=np.exp(rng.normal(scale=.2,size=(Q,P)));Zh=Z*np.exp(rng.normal(scale=.2,size=(Q,P)));N=Z[:,:,None]*rng.normal(size=(Q,P,D));Nh=N+rng.normal(scale=.2,size=N.shape);p=Problem(dict(N=N,Z=Z,Nh=Nh,Zh=Zh,own=np.array(0)),K,False);H,B,L,U,sh,guard,meta=build_h(p);m=P-K;R=np.linalg.cholesky(B[1:,1:]).T
 exact=float('inf')
 for c in itertools.combinations(range(1,P),K-1):
  y=np.ones(P);y[0]=0;y[list(c)]=0;exact=min(exact,float(y@H@y))
 model=Model();model.hideOutput();model.setRealParam('numerics/epsilon',1e-14);model.setRealParam('numerics/feastol',1e-10);model.setRealParam('limits/absgap',1e-12);model.setRealParam('limits/gap',1e-10);model.setRealParam('limits/time',120);model.setBoolParam('constraints/nonlinear/assumeconvex',False);model.setBoolParam('nlp/disable',True)
 y=[model.addVar(vtype='B') for _ in range(P-1)];v=[model.addVar(lb=None,ub=None) for _ in range(P-1)];ep=model.addVar(lb=0.);model.addCons(quicksum(y)==m)
 for i in range(P-1):model.addCons(v[i]==quicksum(float(R[i,j])*y[j] for j in range(P-1)))
 model.addCons(ep>=quicksum(z*z for z in v));model.setObjective(ep);model.optimize();lb=float(model.getDualbound())-sh*m;ub=float(model.getPrimalbound())-sh*m
 assert lb<=exact+1e-9 and abs(ub-exact)<=1e-9,(lb,ub,exact)
 rows.append(dict(P=P,K=K,Q=Q,D=D,exhaustive_minimum=exact,solver_LB=lb,solver_UB=ub,solver_status=str(model.getStatus()),passed=True));print(rows[-1],flush=True)
Path('/tmp/e11_compact/factor_solver_small_checks.json').write_text(json.dumps(rows,indent=2))
