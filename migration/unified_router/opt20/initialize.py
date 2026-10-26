import argparse
import torch
from .setup import ROOT,PARENT,PILOTS
from .model import TileRouter
from ..common import config,save,atomic_json,sha,paired_seed
from ..opt17.data import Events

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;paired_seed(20000+a.layer)
    dest=ROOT/f'initialization/layer{a.layer:02d}.pt'
    if dest.exists():print('exists',dest);return
    _,m=config(ROOT);model=TileRouter(m,a.layer,'Tile15').cuda()
    if a.layer in PILOTS:
        source=PARENT/f'training/layer{a.layer:02d}/Fast35/best.pt'
        model.load_state_dict(torch.load(source,map_location='cuda',weights_only=True)['model'])
        info=dict(source=str(source),source_sha256=sha(source),trees='Opt18 best',events=[])
    else:
        events=Events(m,a.layer);q,k,ids=events.calibration_tensors(20000+a.layer,m['calibration_events'])
        model.set_trees(*model.propose_trees(q,k,projected=True))
        info=dict(source=str(m['parent_experiment']),trees='Train-only projected covariance calibration',events=ids)
    save(dest,dict(model=model.state_dict(),layer=a.layer,shared_by_both_budgets=True,info=info))
    atomic_json(dest.with_suffix('.json'),dict(layer=a.layer,**info,sha256=sha(dest)))
    print('initialized',a.layer,flush=True)

if __name__=='__main__':main()
