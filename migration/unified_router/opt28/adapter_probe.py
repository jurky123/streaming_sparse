import torch,gc
from .setup import ROOT,tag
from .retile import select as old
from .retile_v2 import select as new
from ..opt24.probe import load
from ..opt5.probe import capture
from ..opt26.probe import measure
from ..common import atomic_json
@torch.no_grad()
def main():
 torch.set_num_threads(4);rows=[]
 for h,w in [(480,832),(720,1280),(960,1664)]:
  for chunks in [4,7]:
   for layer in [1,14,24,29]:
    raw,model,pipe,ref,aux=load(h,w,chunks,layer);b64=aux['kv'].state['key'].shape[1]
    for g,k in [(64,64),(128,64),(64,128)]:
     args=(aux['fine'],aux['candidates'],b64,pipe.seg,g,k)
     bg,bo=capture(lambda:old(*args));bg.replay();bo=bo.clone();r=measure(lambda:new(*args),bg,bo,reps=20);assert r['exact'],r
     r.update(shape=tag(h,w,chunks),layer=layer,q=g,k=k);rows.append(r);print(r,flush=True);atomic_json(ROOT/'adapter.json',dict(completed=False,rows=rows));del bg,bo
    del raw,model,pipe,ref,aux,args;gc.collect();torch.cuda.empty_cache()
 atomic_json(ROOT/'adapter.json',dict(completed=True,rows=rows))
if __name__=='__main__':main()
