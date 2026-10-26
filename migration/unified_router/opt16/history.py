"""Explicit owned history snapshot; current KV is always repacked/rebuilt.

Caller must rebuild the snapshot when history content, length/age, model weights,
or grouping trees change. No pointer-only or cross-denoising implicit cache.
"""
from dataclasses import dataclass
import math
import torch
from ..core import part
from .grouping import order,gather
from .route import prepare
from .pipeline import PackedKV

def _segment(k,weight,bias,basis):
    pp=part(k,None,weight,bias);h,b,p,_=pp.key.shape
    key=torch.bmm(pp.key.flatten(1,2),basis[...,:64])
    lc=pp.count.clamp_min(1e-30).log().masked_fill(pp.count<=1e-8,-torch.inf)
    return key,lc,None

@dataclass(frozen=True)
class History:
    k: torch.Tensor
    v: torch.Tensor
    order: torch.Tensor
    parts: tuple
    owner: object

class HistoryBuilder:
    def __init__(self,pipeline,compile=True):
        self.pipeline=pipeline
        self.segment=torch.compile(_segment,fullgraph=True) if compile else _segment

    def pack(self,k,v,start,group):
        pipeline=self.pipeline
        ix=order(k,pipeline.ktree) if group else torch.arange(k.shape[1],device=k.device,dtype=torch.int32)[None].expand(k.shape[0],-1)
        return gather(k,ix),gather(v,ix),ix+start

    def history(self,k,v,group=True):
        if k.shape!=v.shape or k.shape[1]==0 or k.shape[1]%4680:
            raise ValueError('History must contain complete 4680-token chunks with matching K/V')
        par=self.pipeline.parameters;parts=[];keys=[];values=[];indices=[]
        for start in range(0,k.shape[1],4680):
            end=min(start+4680,k.shape[1]);age=math.ceil((k.shape[1]-start)/4680);p=8 if age==1 else 4 if age==2 else 2
            kk,vv,ix=self.pack(k[:,start:end],v[:,start:end],start,group)
            parts.append(self.segment(kk,par[f'history_{p}_weight'],par[f'history_{p}_bias'],par['key_basis']))
            keys.append(kk);values.append(vv);indices.append(ix)
        if not keys:raise ValueError('First chunk must use dense attention; grouped history cannot be empty')
        return History(torch.cat(keys,1),torch.cat(values,1),torch.cat(indices,1),tuple(parts),self)

    def current(self,history,k,v,group=True):
        if history.owner is not self:raise ValueError('History belongs to a different grouping/parameter builder')
        if k.shape!=v.shape or not 0 < k.shape[1] <= 4680:
            raise ValueError('Current K/V must contain one nonempty chunk')
        par=self.pipeline.parameters;nh=history.k.shape[1]
        kk,vv,ix=self.pack(k,v,nh,group)
        pp=self.segment(kk,par['current_weight'],par['current_bias'],par['key_basis'])
        state=prepare((par['key_basis'][...,:64],list(history.parts)+[pp]))
        return PackedKV(torch.cat((history.k,kk),1),torch.cat((history.v,vv),1),torch.cat((history.order,ix),1),state,nh)

    def run(self,q,history,k,v,group_q=True,group_k=True):
        return self.pipeline.run(q,self.current(history,k,v,group_k),group_q)
