import argparse,numpy as np
from pathlib import Path
from .solver import solve
p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--limit',type=float,required=True);a=p.parse_args();path=Path(a.path);data=dict(np.load(path));solve(data,int(data['Z'].shape[1]*.1),int(data['own']),[data['mass'],data['risk']],path.parent/'solve',limit=a.limit)
