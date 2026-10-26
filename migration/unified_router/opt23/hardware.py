"""Warm, uncaptured execution range for Nsight Compute hardware counters."""
import argparse,torch
from .setup import ROOT,TRAINING,tag
from .pipeline import Pipeline
from ..common import config
from ..opt20.model import TileRouter
from ..opt16.pipeline import GroupedRouter
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--height',type=int,default=480);p.add_argument('--width',type=int,default=832);p.add_argument('--chunks',type=int,default=4);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw=torch.load(ROOT/'capture'/tag(a.height,a.width,a.chunks)/'layer24.pt',map_location='cuda',weights_only=False);_,m=config(TRAINING)
 model=TileRouter(m,24,'Tile20').cuda().eval();model.load_state_dict(torch.load(TRAINING/'training/layer24/Tile20/best.pt',map_location='cuda',weights_only=True)['model']);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False)
 pipe=Pipeline(model,raw,raw['chunk_tokens'])
 for _ in range(5):y=pipe.run(raw['q'],raw['kc'],raw['vc'])
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
 y=pipe.run(raw['q'],raw['kc'],raw['vc']);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 assert torch.isfinite(y).all()
if __name__=='__main__':main()
