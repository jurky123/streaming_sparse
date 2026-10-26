import argparse,torch
from .setup import ROOT
from ..common import config,save,atomic_json
from .data import Events
from .model import JointGroupedRouter

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);a=p.parse_args()
    _,m=config(ROOT);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    folder=ROOT/'calibration';folder.mkdir(exist_ok=True)
    atomic_json(folder/f'layer{a.layer:02d}.status.json',dict(status='loading'))
    events=Events(m,a.layer);q,k,ids=events.calibration_tensors(17000+a.layer,m['calibration_events'])
    atomic_json(folder/f'layer{a.layer:02d}.status.json',dict(status='fitting',events=ids))
    model=JointGroupedRouter(m,a.layer,'Fast35').cuda();qt,kt=model.propose_trees(q,k);model.set_trees(qt,kt)
    trees={key:value for key,value in model.state_dict().items() if '_tree_'in key}
    save(folder/f'layer{a.layer:02d}.pt',dict(trees=trees,layer=a.layer,depth=m['tree_depth'],events=ids,split='train'))
    atomic_json(folder/f'layer{a.layer:02d}.status.json',dict(status='completed',events=ids,train_only=True))
    print('calibrated',a.layer,len(ids),flush=True)

if __name__=='__main__':main()
