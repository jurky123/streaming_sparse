import argparse,json,numpy as np
from pathlib import Path
from ..e11_oracle.solver import solve,objective
from ..e11_oracle.contributions import diagnostic
p=argparse.ArgumentParser();p.add_argument('path');a=p.parse_args();path=Path(a.path);data=dict(np.load(path));root=path.parent/path.stem
r=solve(data,int(data['K']),int(data['own']),[data['mass'],data['risk']],root,limit=900)
mask=np.zeros(data['Z'].shape[1],bool);mask[r['mask']]=True;rows={}
for method,m in [('mass',data['mass']),('risk',data['risk']),('opt',mask)]:
 arrays,met=diagnostic(data,m);np.savez_compressed(root/f'{method}_D02.npz',mask=m,**arrays);rows[method]=met
(root/'metrics.json').write_text(json.dumps(rows,indent=2));print(path,r['status'],r['LB'],r['UB'],flush=True)
