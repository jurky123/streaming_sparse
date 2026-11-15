import json,numpy as np
from pathlib import Path
from .solver import solve,exhaustive
ROOT=Path('/tmp/e11_results')
def main():
 path=ROOT/'contributions/horse_field_s101_c2_l1_s0.npz';full=dict(np.load(path));own=int(full['own']);rows=[]
 for P,K,Q,d in [(12,2,4,8),(16,3,4,128),(24,4,64,8),(12,2,64,128)]:
  pages=np.r_[own,np.delete(np.arange(full['Z'].shape[1]),own)[:P-1]]
  data={n:full[n][:Q,pages,:d] if n in ('N','Nh') else full[n][:Q,pages] for n in ('N','Z','Nh','Zh')}
  val,mask,count=exhaustive(data,K,0);r=solve(data,K,0,[np.arange(P)<K],ROOT/'validation_final'/f'real_P{P}_K{K}_Q{Q}_d{d}',limit=900,checkpoints=(5,60,300,900))
  passed=abs(r['UB']-val)<=max(1e-10,1e-6*abs(val)) and r['LB']<=val+1e-10 and r['status']=='GLOBAL_TOL'
  rows.append(dict(kind='real_reduced',P=P,K=K,Q=Q,d=d,pages=pages.tolist(),exhaustive_value=val,sets=count,passed=passed,result=r));(ROOT/'real_small_tests.json').write_text(json.dumps(rows,indent=2));print('REAL',P,Q,d,passed,flush=True)
 assert all(z['passed'] for z in rows)
if __name__=='__main__':main()
