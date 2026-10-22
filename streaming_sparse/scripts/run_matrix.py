"""Launch sequential router experiments; no backbone rerun required.

Run three seeds at default settings, optionally sweep query tiles and alpha.
All generated configurations and commands are written before execution.
"""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='streaming_sparse/train/configs/p2.json')
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--tiles',nargs='+',type=int,default=[16])
    p.add_argument('--alphas',nargs='+',type=float,default=[0.0])
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    base=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    commands=[]
    for seed,tile,alpha in itertools.product(args.seeds,args.tiles,args.alphas):
        cfg=json.loads(json.dumps(base))
        cfg['seed']=seed
        cfg.setdefault('router',{}).update(query_tile=tile,alpha=alpha)
        path=output/f'seed{seed}_tile{tile}_alpha{alpha}'
        path.mkdir(parents=True,exist_ok=True)
        config=path/'config.json'
        config.write_text(json.dumps(cfg,indent=2))
        commands.append([sys.executable,'-m','streaming_sparse.train.train_router','--config',str(config),
                         '--data',args.data,'--output',str(path)])
    (output/'commands.json').write_text(json.dumps(commands,indent=2))
    if not args.prepare_only:
        for command in commands:
            subprocess.run(command,check=True)


if __name__=='__main__':
    main()
