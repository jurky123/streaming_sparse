"""Training-free, fixed-size per-clean-chunk grouping. No future queries."""
import math
import torch

PAGE=64
CHUNK=4608


@torch.no_grad()
def balanced_order(k,v,weight=0.,iterations=4):
    """[H, 4608, D] -> [H, 4608] permutation; balanced 3*3*2*2*2 tree.

    Normalize K and centered V by per-head scalar RMS so lambda controls the
    squared-distance contribution. Each split estimates a leading PC by power
    iteration, starts on the highest-variance dimension, then sorts stably.
    """
    h,n,d=k.shape
    assert n==CHUNK,(k.shape,CHUNK)
    k=k.float()-k.float().mean(1,keepdim=True)
    k=k/k.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)
    if weight:
        v=v.float()-v.float().mean(1,keepdim=True)
        v=v/v.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)
        x=torch.cat((k,v*math.sqrt(weight)),dim=-1)
    else:x=k
    ids=torch.arange(n,device=k.device).expand(h,n).clone()
    groups=1
    for factor in [3,3,2,2,2]:
        size=n//groups
        x=x.reshape(h*groups,size,-1)
        ids=ids.reshape(h*groups,size)
        xc=x-x.mean(1,keepdim=True)
        var=xc.square().mean(1)
        direction=torch.zeros_like(var).scatter_(1,var.argmax(-1,keepdim=True),1.)
        for _ in range(iterations):
            projection=torch.bmm(xc,direction.unsqueeze(-1))
            direction=torch.bmm(xc.transpose(1,2),projection).squeeze(-1)
            direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        score=torch.bmm(xc,direction.unsqueeze(-1)).squeeze(-1)
        order=score.argsort(dim=-1,stable=True)
        x=x.gather(1,order.unsqueeze(-1).expand_as(x))
        ids=ids.gather(1,order)
        groups*=factor
    assert groups==72
    return ids.reshape(h,n)


def reorder(x,order):
    return x.gather(1,order.long().unsqueeze(-1).expand(-1,-1,x.shape[-1]))


@torch.no_grad()
def history_order(k,v,weight):
    h,n,d=k.shape
    if weight is None:return torch.arange(n,device=k.device).expand(h,n).clone()
    return torch.cat([balanced_order(k[:,s:s+CHUNK],v[:,s:s+CHUNK],weight)+s
                      for s in range(0,n,CHUNK)],1)


def page_stats(k,v):
    h,n,d=k.shape
    kp=k.float().reshape(h,n//PAGE,PAGE,d)
    vp=v.float().reshape(h,n//PAGE,PAGE,d)
    km=kp.mean(2);vm=vp.mean(2)
    return km,vm,(kp-km[:,:,None]).square().mean(2),(vp-vm[:,:,None]).square().sum(-1).mean(2)


def page_budget(pages,ratio=.2,tail=False,variance=False):
    tokens=int(pages*PAGE*ratio)
    # Our dense summary GEMMs evaluate ALL page candidates before masking.
    # Charge 1 equivalent per page for mean KV, 2 for mean+diagonal variance
    # (conservative: 3 dot products rather than 2 per ordinary attention term).
    summary_cost=pages*(2 if variance else 1) if tail else 0
    selected=max(0,(tokens-summary_cost)//PAGE) if tail else tokens//PAGE
    assert selected*PAGE+summary_cost<=tokens
    return min(pages,selected)
