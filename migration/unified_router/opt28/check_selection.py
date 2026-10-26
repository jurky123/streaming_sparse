import torch
from .retile import select
from ..opt24.probe import load
@torch.no_grad()
def main():
 torch.set_num_threads(4)
 raw,model,pipe,ref,aux=load(720,1280,7,24)
 for g,k in [(32,128),(64,128),(128,128)]:
  ids=select(aux['fine'],aux['candidates'],aux['kv'].state['key'].shape[1],pipe.seg,g,k);torch.cuda.synchronize();b=7*((pipe.seg+k-1)//k)
  print(g,k,'b',b,'range',int(ids.min()),int(ids.max()),'sorted',bool((ids.diff(dim=-1)>0).all()),'invalid',int(((ids<0)|(ids>=b)).sum()),flush=True)
if __name__=='__main__':main()
