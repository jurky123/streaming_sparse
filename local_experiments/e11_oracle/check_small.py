import argparse,json,numpy as np
from pathlib import Path
from .solver import solve,exhaustive
ROOT=Path('/tmp/e11_results')
def main():
 rng=np.random.default_rng(17);rows=[]
 for P,K,Q,d in [(12,2,4,8),(16,3,4,128),(24,4,64,8),(12,2,64,128)]:
  Z=np.exp(rng.normal(scale=.3,size=(Q,P)));Zh=Z*np.exp(rng.normal(scale=.2,size=(Q,P)));N=Z[:,:,None]*rng.normal(size=(Q,P,d));Nh=N+rng.normal(scale=.2,size=N.shape)
  data=dict(N=N,Z=Z,Nh=Nh,Zh=Zh);val,mask,count=exhaustive(data,K,0)
  result=solve(data,K,0,[np.arange(P)<K],ROOT/'validation_final'/f'synthetic_P{P}_K{K}_Q{Q}_d{d}',limit=900,checkpoints=(5,60,300,900))
  passed=abs(result['UB']-val)<=max(1e-10,1e-6*abs(val)) and result['LB']<=val+1e-10 and result['status']=='GLOBAL_TOL'
  rows.append(dict(kind='synthetic',P=P,K=K,Q=Q,d=d,exhaustive_value=val,sets=count,passed=passed,result=result));print('SMALL',P,Q,d,passed,flush=True)
  (ROOT/'small_tests.json').write_text(json.dumps(rows,indent=2))
 assert all(r['passed'] for r in rows)
if __name__=='__main__':main()
