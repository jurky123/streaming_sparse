"""Selection-based query permutations. No selected page is added or removed."""
import torch
from functools import lru_cache

@lru_cache(maxsize=32)
def signs(b,bits,device):
 p=torch.arange(b,device=device,dtype=torch.int64)[:,None];s=torch.arange(bits,device=device,dtype=torch.int64)[None,:]
 v=((p+1)*1103515245+(s+1)*2654435761);v=(v^(v>>16))*2246822519;v=(v^(v>>13))&0x7fffffff
 return ((v&1)*2-1).half()

def order(mask,method):
 h,n,b=mask.shape
 if method.startswith("BitsetFast"):
  from .bitset_fast import bitset_order
  return bitset_order(mask,method)
 if method.startswith("Bitset"):
  from .bitset_orders import bitset_order
  return bitset_order(mask,method)
 if method.startswith(('LocalAnchor','Greedy')):
  from .local_orders import local_order
  return local_order(mask,method)
 if method=='Identity':return torch.arange(n,device=mask.device).expand(h,n).contiguous()
 if method.startswith('Anchor')or method.startswith('Kmeans'):
  k=int(method.removeprefix('Anchor').removeprefix('Kmeans').split('_')[0]);iters=int(method.split('_')[1])if '_'in method else 0
  x=mask.half();indices=torch.arange(k,device=mask.device)*n//k;centers=x[:,indices].contiguous()
  for i in range(iters+1):
   affinity=2*torch.bmm(x,centers.transpose(1,2)).float()-centers.float().square().sum(-1)[:,None]
   labels=affinity.argmax(-1)
   if i<iters:
    sums=torch.zeros((h,k,b),device=mask.device,dtype=torch.float32);sums.scatter_add_(1,labels[:,:,None].expand(h,n,b),x.float())
    counts=torch.zeros((h,k),device=mask.device);counts.scatter_add_(1,labels,torch.ones_like(labels,dtype=torch.float32));centers=(sums/counts.clamp_min(1)[:,:,None]).half()
  return labels.argsort(dim=-1,stable=True)
 if method.startswith('Sim'):
  bits=int(method[3:]);proj=torch.matmul(mask.half(),signs(b,bits,mask.device)).float();positive=proj>proj.mean(1,keepdim=True)
  key=(positive.long()*(1<<torch.arange(bits,device=mask.device))).sum(-1)
 elif method.startswith('Pivot'):
  bits=int(method[5:]);freq=mask.float().mean(1);pivots=(-(freq-.5).abs()).topk(bits,dim=-1).indices
  chosen=mask.gather(2,pivots[:,None,:].expand(h,n,bits));key=(chosen.long()*(1<<torch.arange(bits,device=mask.device))).sum(-1)
 elif method=='Mean':
  p=torch.arange(b,device=mask.device,dtype=torch.float32);key=(mask*p).sum(-1)/mask.sum(-1).clamp_min(1)
 elif method.startswith('Minhash'):
  seed=int(method[7:]);p=torch.arange(b,device=mask.device,dtype=torch.int64);v=((p+1)*1103515245+seed*2654435761)&0x7fffffff;v=(v^(v>>16))%65521
  key=torch.where(mask,v[None,None],2**31-1).min(-1).values
 else:raise ValueError(method)
 return key.argsort(dim=-1,stable=True)
