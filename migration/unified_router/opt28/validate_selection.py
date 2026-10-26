"""Independent PyTorch check of shared top-k; includes partial Q/K tails."""
import torch,math,gc
from .setup import ROOT,tag
from .retile import select
from ..opt24.probe import load
from ..common import atomic_json
@torch.no_grad()
def check(fine,candidates,ids,b64,seg,g,k):
 h,n,c=fine.shape;p64=math.ceil(seg/64);pk=math.ceil(seg/k);chunks=b64//p64;gaps=[]
 for head in [0,5,11]:
  for tile in [0,ids.shape[1]//2,ids.shape[1]-1]:
   score=torch.zeros(b64,device=fine.device);start=tile*g;end=min(n,start+g)
   for block in range(start//32,math.ceil(end/32)):
    lo=max(start,block*32);hi=min(end,(block+1)*32)
    prob=fine[head,lo:hi].float().softmax(-1).sum(0)
    score.scatter_add_(0,candidates[head,block].long(),prob)
   if k==128:
    padded=torch.zeros((chunks,pk*2),device=fine.device);padded[:,:p64]=score.reshape(chunks,p64);score=padded.reshape(chunks,pk,2).sum(-1).flatten()
   chosen=ids[head,tile].long();selected=score[chosen];score[chosen]=-float('inf');gap=float(score.max()-selected.min());assert gap<1e-5,(head,tile,g,k,gap);gaps.append(max(0.,gap))
 return max(gaps)
@torch.no_grad()
def main():
 torch.set_num_threads(4);rows=[]
 for h,w in [(480,832),(720,1280),(960,1664)]:
  for chunks in [4,7]:
   for layer in [1,14,24,29]:
    raw,model,pipe,ref,aux=load(h,w,chunks,layer);b64=aux['kv'].state['key'].shape[1];checks=[]
    for g,k in [(16,64),(32,64),(64,64),(128,64),(256,64),(16,128),(32,128),(64,128),(128,128),(256,128)]:
     ids=select(aux['fine'],aux['candidates'],b64,pipe.seg,g,k);b=chunks*math.ceil(pipe.seg/k)
     assert bool(((ids>=0)&(ids<b)).all()) and bool((ids[:,:,1:]>ids[:,:,:-1]).all())
     gap=check(aux['fine'],aux['candidates'],ids,b64,pipe.seg,g,k);checks.append(dict(q=g,k=k,max_topk_violation=gap))
    row=dict(shape=tag(h,w,chunks),layer=layer,checks=checks);rows.append(row);print(row,flush=True);atomic_json(ROOT/'selection_verification.json',dict(completed=False,rows=rows))
    del raw,model,pipe,ref,aux,ids;gc.collect();torch.cuda.empty_cache()
 atomic_json(ROOT/'selection_verification.json',dict(completed=True,rows=rows))
if __name__=='__main__':main()
