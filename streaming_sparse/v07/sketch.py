"""Deployable joint KV prototypes: no teacher/dense/history statistics inputs."""
import math
import torch
from torch import nn
from .exact import pages,geometry,page_costs,partition_hash


class Sketch(nn.Module):
    def __init__(self,prototypes=4,key_only=False,variance=False,page_size=64):
        super().__init__();self.prototypes=prototypes;self.key_only=key_only;self.variance=variance;self.page_size=page_size
        width=128 if key_only else 256
        if prototypes>1:
            self.weight=nn.Parameter(torch.randn(36,prototypes,width)*.01)
            self.bias=nn.Parameter(torch.zeros(36,prototypes))
        else:
            self.register_parameter('weight',None);self.register_parameter('bias',None)

    def forward(self,k,v,pair,permutation=None):
        if permutation is None:permutation=torch.arange(len(k),device=k.device)
        kk,vv,valid=pages(k[permutation],v[permutation],self.page_size)
        inputs=kk if self.key_only else torch.cat((kk,vv),-1)
        if self.prototypes==1:a=torch.ones((*kk.shape[:2],1),device=k.device)
        else:a=torch.softmax(torch.nn.functional.linear(inputs,self.weight[pair],self.bias[pair]),-1)
        a=a*valid[...,None];count=a.sum(1)
        denominator=count.clamp_min(1e-30)[...,None]
        kb=torch.einsum('btp,btd->bpd',a,kk)/denominator
        vb=torch.einsum('btp,btd->bpd',a,vv)/denominator
        variance=(torch.einsum('btp,btd->bpd',a,kk.square())/denominator-kb.square()).clamp_min(0) if self.variance else None
        return dict(count=count,key=kb,value=vb,variance=variance,valid=count>0,
                    costs=page_costs(len(k),self.page_size,k.device),indices=permutation,
                    partition_hash=partition_hash(permutation,self.page_size),pair=pair,history_tokens=len(k))

    def estimate(self,q,state,current,history_only=False):
        logits=torch.einsum('qd,bpd->qbp',q.float(),state['key'])/math.sqrt(128)
        logits=logits+state['count'].clamp_min(1e-30).log()[None]
        clipped=0
        if self.variance:
            correction=torch.einsum('qd,bpd->qbp',q.float().square(),state['variance'])/(2*128)
            clipped=int((correction.detach()>20).sum());logits=logits+correction.clamp(max=20)
        logits=logits.masked_fill(~state['valid'][None],-torch.inf)
        L=torch.logsumexp(logits,-1)
        mu=torch.einsum('qbp,bpd->qbd',logits.softmax(-1),state['value'])
        if not torch.isfinite(L).all() or not torch.isfinite(mu).all():raise FloatingPointError('Nonfinite prototype estimates')
        result=geometry(L,mu,current[0],current[1],history_only)
        result['variance_clipped']=clipped
        return result

    def config(self):return dict(prototypes=self.prototypes,key_only=self.key_only,variance=self.variance,page_size=self.page_size)
