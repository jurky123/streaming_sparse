from collections import OrderedDict
import math
from pathlib import Path
import torch
from ..cache_grouping import history_order,reorder
from .model import prepare

SOURCE=Path(__file__).resolve().parents[2]/'artifacts/cache_research_20261029/capture'


def records(split):
    scenes=['sailboat'] if split in ['train','val'] else ['skateboard','dog']
    result=[]
    for scene in scenes:
        for path in sorted((SOURCE/scene/'qkv').glob('*_s?.pt')):
            chunk=int(path.name[1:3])
            if split=='train' and chunk==40:continue
            if split=='val' and chunk!=40:continue
            result.append(path)
    return result


@torch.no_grad()
def teacher(q,k,v,kc,vc):
    h,n,d=k.shape;p=n//64
    logits=(q@k.transpose(1,2)/math.sqrt(d)).reshape(h,q.shape[1],p,64)
    z=logits.logsumexp(-1);values=torch.einsum('hqpn,hpnd->hqpd',logits.softmax(-1),v.reshape(h,p,64,d))
    cl=q@kc.transpose(1,2)/math.sqrt(d)
    z=torch.cat((z,cl.logsumexp(-1)[:,:,None]),-1)
    values=torch.cat((values,(cl.softmax(-1)@vc)[:,:,None]),2)
    output=(z.softmax(-1)[:,:,:,None]*values).sum(2)
    return z,values,output


class Bank:
    def __init__(self,max_history=3,max_events=4,queries=64):
        self.hist=OrderedDict();self.events=OrderedDict();self.max_history=max_history;self.max_events=max_events;self.queries=queries

    @torch.no_grad()
    def get(self,path):
        key=str(path)
        if key in self.events:
            self.events.move_to_end(key);return self.events[key]
        hp=path.with_name(path.name[:-6]+'_history.pt');hk=str(hp)
        if hk not in self.hist:
            data=torch.load(hp,map_location='cpu',weights_only=True)
            k=data['k'].cuda();v=data['v'].cuda();order=history_order(k,v,0.)
            k=reorder(k,order).float();v=reorder(v,order).float()
            self.hist[hk]=dict(state=prepare(k,v),k=k,v=v,layer=data['layer'],heads=data['heads'])
            while len(self.hist)>self.max_history:self.hist.popitem(last=False)
        else:self.hist.move_to_end(hk)
        hist=self.hist[hk]
        data=torch.load(path,map_location='cpu',weights_only=True)
        q=data['q'].cuda().float();kc=data['k'].cuda().float();vc=data['v'].cuda().float()
        ids=torch.linspace(0,q.shape[1]-1,self.queries,device='cuda').long()
        route=q.reshape(q.shape[0],-1,64,q.shape[-1]).mean(2)[:,ids//64];q=q[:,ids]
        z,val,target=teacher(q,hist['k'],hist['v'],kc,vc)
        event=dict(hist=hist,q=q,route=route,kc=kc,vc=vc,z=z,values=val,target=target,
                   case=path.parent.parent.name,chunk=data['chunk'],step=int(path.stem[-1]),path=key)
        self.events[key]=event
        while len(self.events)>self.max_events:self.events.popitem(last=False)
        return event
