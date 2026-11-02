"""Learned within-hard-group K/V pooling and V-aware equal-budget swaps."""
from collections import OrderedDict
import math
from pathlib import Path
import torch
from torch import nn
from .response_hard import balanced, features
from .cache_grouping import reorder
from .learned_cache.data import teacher
from .learned_cache.model import initial_mask, output_for_masks

DICTIONARY=Path('artifacts/response_hard_20261030/dictionary.pt')

class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Embedding(30,8);self.head=nn.Embedding(12,8)
        self.net=nn.Sequential(nn.Linear(304,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,h,learned=True,which='both'):
        k,v,x=h['gk'],h['gv'],h['x'];heads,p,r,n,d=k.shape
        if not learned:return k.mean(3),v.mean(3),torch.full((heads,p,r),float(n),device=k.device)
        emb=torch.cat((self.layer.weight[h['layer']].expand(heads,-1),self.head.weight[h['heads']]),-1)
        emb=emb[:,None,None,None].expand(heads,p,r,n,16)
        logits=3*torch.tanh(self.net(torch.cat((x,emb),-1))/3)
        weights=logits.softmax(3)
        pk=(weights[...,0,None]*k).sum(3) if which!='v' else k.mean(3)
        pv=(weights[...,1,None]*v).sum(3) if which!='k' else v.mean(3)
        return pk,pv,torch.full((heads,p,r),float(n),device=k.device)


class Bank:
    def __init__(self,prototypes,max_history=6,max_events=24):
        self.prototypes=prototypes;self.hist=OrderedDict();self.events=OrderedDict()
        self.max_history=max_history;self.max_events=max_events
        self.dictionary={k:v.cuda() for k,v in torch.load(DICTIONARY,weights_only=True).items()}

    @torch.no_grad()
    def get(self,path):
        key=str(path)
        if key in self.events:self.events.move_to_end(key);return self.events[key]
        hp=path.with_name(path.name[:-6]+'_history.pt');hk=str(hp)
        if hk not in self.hist:
            data=torch.load(hp,weights_only=True);k=data['k'].cuda().float();v=data['v'].cuda().float()
            orders=[];xs=[];layer=data['layer']
            for start in range(0,k.shape[1],4608):
                x=features(k[:,start:start+4608],v[:,start:start+4608],self.dictionary[layer],'response')
                order=balanced(x,[3,3,2,2,2]);xp=reorder(x,order)
                if self.prototypes==4:
                    inner=balanced(xp.reshape(-1,64,32),[2,2]).reshape(3,-1,64)
                    order=order.reshape(3,-1,64).gather(2,inner).flatten(1)
                orders.append(order+start);xs.append(reorder(x,order))
            order=torch.cat(orders,1)
            assert torch.equal(order.sort(-1).values,torch.arange(k.shape[1],device=k.device).expand_as(order))
            k=reorder(k,order);v=reorder(v,order);r=self.prototypes
            gk=k.reshape(3,-1,r,64//r,128);gv=v.reshape_as(gk)
            def centered(t):
                z=t-t.mean(3,keepdim=True)
                return z/z.square().mean((3,4),keepdim=True).sqrt().clamp_min(1e-5)
            gx=torch.cat(xs,1).reshape(3,-1,r,64//r,32)
            x=torch.cat((centered(gk),centered(gv),centered(gx)),-1)
            self.hist[hk]=dict(k=k,v=v,gk=gk,gv=gv,x=x,layer=layer,heads=data['heads'])
            while len(self.hist)>self.max_history:self.hist.popitem(last=False)
        else:self.hist.move_to_end(hk)
        h=self.hist[hk];data=torch.load(path,weights_only=True)
        q=data['q'].cuda().float();kc=data['k'].cuda().float();vc=data['v'].cuda().float()
        ids=torch.linspace(0,q.shape[1]-1,64,device='cuda').long()
        route=q.reshape(3,-1,64,128).mean(2)[:,ids//64];q=q[:,ids]
        z,values,target=teacher(q,h['k'],h['v'],kc,vc)
        event=dict(h=h,q=q,route=route,kc=kc,vc=vc,z=z,values=values,target=target,
                   scene=path.parent.parent.name,chunk=data['chunk'],step=int(path.stem[-1]))
        self.events[key]=event
        while len(self.events)>self.max_events:self.events.popitem(last=False)
        return event


def actions(z,values,kind):
    """Same five actions: KEEP plus 2x2 swaps. Only candidate generation differs."""
    mask=initial_mask(z);prob=z.softmax(-1);mass=(prob*mask).sum(-1)
    selected=(prob[:,:,:,None]*mask[:,:,:,None]*values).sum(2)/mass[:,:,None]
    full=(prob[:,:,:,None]*values).sum(2)
    error=(selected-full).square().mean(-1)
    if kind=='boundary':
        remove=z[...,:-1].masked_fill(~mask[...,:-1],torch.inf).topk(2,largest=False).indices
        add=z[...,:-1].masked_fill(mask[...,:-1],-torch.inf).topk(2).indices
    else:
        delta=values[:,:,:-1]-selected[:,:,None]
        pm=prob[...,:-1]
        added=selected[:,:,None]+(pm/(mass[:,:,None]+pm))[:,:,:,None]*delta
        dropped=selected[:,:,None]-(pm/(mass[:,:,None]-pm).clamp_min(1e-8))[:,:,:,None]*delta
        addgain=error[:,:,None]-(added-full[:,:,None]).square().mean(-1)
        dropgain=error[:,:,None]-(dropped-full[:,:,None]).square().mean(-1)
        add=addgain.masked_fill(mask[...,:-1],-torch.inf).topk(2).indices
        remove=dropgain.masked_fill(~mask[...,:-1],-torch.inf).topk(2).indices
    ri=remove.repeat_interleave(2,-1);aj=add.repeat(1,1,2)
    mi=prob.gather(-1,ri);mj=prob.gather(-1,aj)
    vi=values.gather(2,ri[:,:,:,None].expand(-1,-1,-1,values.shape[-1]))
    vj=values.gather(2,aj[:,:,:,None].expand_as(vi))
    out=(mass[:,:,None,None]*selected[:,:,None]-mi[:,:,:,None]*vi+mj[:,:,:,None]*vj)/(mass[:,:,None]-mi+mj).clamp_min(1e-8)[:,:,:,None]
    errors=torch.cat((error[:,:,None],(out-full[:,:,None]).square().mean(-1)),-1)
    page=torch.arange(z.shape[-1],device=z.device)
    swaps=(mask[:,:,None]&(page!=ri[:,:,:,None]))|(page==aj[:,:,:,None])
    masks=torch.cat((mask[:,:,None],swaps),2)
    assert (masks.sum(-1)==mask.sum(-1)[:,:,None]).all() and masks[...,-1].all()
    return masks,errors.argmin(-1),errors
