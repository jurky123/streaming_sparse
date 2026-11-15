import json,itertools,numpy as np
from pathlib import Path
from .solver import setup,objective
from .contributions import diagnostic
ROOT=Path('/tmp/e11_results');rows=[]
for path in sorted((ROOT/'contributions').glob('*.npz')):
 data=dict(np.load(path));P=data['Z'].shape[1];own=int(data['own']);K=int(P*.1);free,a,b,dn,dz,lo,hi,od,cd=setup(data,K,own)
 assert np.allclose(od,data['dense_heads'][int(data['head'])],rtol=1e-12,atol=1e-12)
 errors=[]
 for name in ('mass','risk'):
  mask=data[name];assert mask.sum()==K and mask[own];y=1/np.where(mask[None],data['Z'],data['Zh']).sum(1);w=y[:,None]*mask[free];t=a*y[:,None]+np.einsum('qpd,qp->qd',dn,w)-od
  assert (y>=lo*(1-1e-12)).all() and (y<=hi*(1+1e-12)).all()
  err=abs(float(np.square(t).sum()/cd)-objective(data,mask));assert err<=1e-12;arrays,met=diagnostic(data,mask);errors.append(dict(method=name,formulation_objective_abs_error=err,identity_max_abs=met['identity_max_abs']))
 # Both boundary budgets: direct masks, no [-0:] indexing.
 for count in (1,P):
  *_,l,u,o,c=setup(data,count,own);assert np.allclose(l,u,rtol=1e-12,atol=0)
 rows.append(dict(instance=path.stem,passed=True,P=P,K=K,checks=errors))
(ROOT/'invariants.json').write_text(json.dumps(rows,indent=2));print('PASS',len(rows))
