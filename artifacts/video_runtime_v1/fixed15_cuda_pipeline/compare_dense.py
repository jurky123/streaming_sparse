import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('latents');p.add_argument('output',type=Path)
args=p.parse_args()
torch.set_num_threads(2)
dense=torch.load('/tmp/jzk_fixed15_release_dense/latents.pt',weights_only=True).float()
old,new=(torch.load(args.latents,weights_only=True)[i].float() for i in (0,1))
def distance(x,ref):
    delta=x-ref
    return dict(nmse=float(delta.square().sum()/ref.square().sum()),
                cosine=float(torch.nn.functional.cosine_similarity(x.flatten(),ref.flatten(),dim=0)),
                chunk_nmse=[float(e.square().sum()/r.square().sum()) for e,r in zip(delta.chunk(9,dim=1),ref.chunk(9,dim=1))])
r=dict(base_dense=distance(old,dense),candidate_dense=distance(new,dense),candidate_base=distance(new,old))
args.output.write_text(json.dumps(r,indent=2));print({k:v['nmse'] for k,v in r.items()})
