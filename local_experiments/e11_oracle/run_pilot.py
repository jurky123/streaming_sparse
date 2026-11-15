import argparse,json,numpy as np
from pathlib import Path
from .solver import solve
from .contributions import diagnostic,selected_check
ROOT=Path('/tmp/e11_results')
def main():
 p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--limit',type=float,default=900);a=p.parse_args();path=Path(a.path);data=dict(np.load(path));own=int(data['own']);K=int(data['Z'].shape[1]*.1);root=ROOT/'pilot'/path.stem
 r=solve(data,K,own,[data['mass'],data['risk']],root,limit=a.limit)
 opt=np.zeros(data['Z'].shape[1],bool);opt[r['mask']]=True;rows={}
 for name,mask in [('O-MASS',data['mass']),('O-RISK',data['risk']),('O-OPT',opt)]:
  arrays,metrics=diagnostic(data,mask);np.savez_compressed(root/(name+'_D02.npz'),mask=mask,**arrays)
  metrics.update(selected_check(ROOT/'snapshots'/(path.stem+'.pt'),mask,data));metrics.update(count=int(mask.sum()),own=bool(mask[own]),selected_pages=np.where(mask)[0].tolist(),gap_to_optimum_interval=[metrics['nmse']-r['UB'],metrics['nmse']-r['LB']]);rows[name]=metrics
 (root/'L1.json').write_text(json.dumps(rows,indent=2));print(path.stem,r['status'],r['LB'],r['UB'],flush=True)
if __name__=='__main__':main()
