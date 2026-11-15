import numpy as np,itertools,json
from pathlib import Path
from .core import Problem
from .relaxation import build_h,qp_relax,soc_relax
root=Path('/tmp/e11_compact');rng=np.random.default_rng(17);full=dict(np.load('/tmp/e11_results/contributions/horse_field_s101_c5_l15_s0.npz'));rows=[]
for kind in ('synthetic','real_subset'):
 for P,K,Q,D in [(12,2,4,8),(16,3,4,128),(24,4,64,8),(12,2,64,128)]:
  if kind=='synthetic':
   Z=np.exp(rng.normal(scale=.2,size=(Q,P)));Zh=Z*np.exp(rng.normal(scale=.2,size=(Q,P)));N=Z[:,:,None]*rng.normal(size=(Q,P,D));Nh=N+rng.normal(scale=.2,size=N.shape);data=dict(N=N,Z=Z,Nh=Nh,Zh=Zh,own=np.array(0))
  else:
   ids=np.r_[int(full['own']),np.delete(np.arange(410),int(full['own']))[:P-1]];data={n:full[n][:Q,ids,:D] if n in ('N','Nh') else full[n][:Q,ids] for n in ('N','Z','Nh','Zh')};data['own']=np.array(0)
  p=Problem(data,K=K,require_full=False);x=np.arange(P)<K;p.data.update(mass=x,risk=x);H,B,L,U,shift,guard,meta=build_h(p);q=qp_relax(p,H,B,shift,guard,maxiter=2000);s=soc_relax(p,U,guard,initial=q['point'],maxiter=200);mins=np.full(3,np.inf);count=0
  for choice in itertools.combinations(range(1,P),K-1):
   x=np.zeros(P,bool);x[0]=True;x[list(choice)]=True;y=(~x).astype(float);E,z,j=p.state(x);v=np.array([y@H@y,np.sum(np.square(E).sum(1)/(z*U))/p.C,p.value(x)]);assert v[0]<=v[1]+1e-10 and v[1]<=v[2]+1e-10;assert np.all(z>=L) and np.all(z<=U);mins=np.minimum(mins,v);count+=1
  assert q['LB']<=mins[0]+1e-10 and s['LB']<=mins[1]+1e-10
  rows.append(dict(kind=kind,P=P,K=K,Q=Q,D=D,sets=count,enumeration_status='EXHAUSTIVE_CHECKED',min_MIQP_surrogate=mins[0],min_SOC_surrogate=mins[1],min_true_J=mins[2],qp_tangent_LB=q['LB'],soc_tangent_LB=s['LB'],passed=True));(root/'lower_bound_small_checks.json').write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True)
