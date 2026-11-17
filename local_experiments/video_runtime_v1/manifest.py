"""Build a reproducible prompt x noise x method manifest; no hidden prompt selection."""
import argparse,json
from pathlib import Path
from .queue import METHODS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);p.add_argument('--cases',nargs='+',default=['city_walk','kitchen_cook']);p.add_argument('--seeds',type=int,nargs='+',default=[701]);p.add_argument('--chunks',type=int,choices=[9,41],default=9);p.add_argument('--output',required=True);a=p.parse_args()
    rows=[dict(case=case,noise_seed=seed,method=method,chunks=a.chunks) for case in a.cases for seed in a.seeds for method in a.methods]
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_text())!=rows:raise ValueError('Existing manifest differs')
    path.write_text(json.dumps(rows,indent=2)+'\n');print(len(rows),'jobs',path)
