"""Distill shared normalized page mass, with optional diagonal query variance."""
import math
import torch
from torch import nn
from .tile_router import Anchored,Events as TileEvents,objective as previous_objective
from .learned_cache.model import initial_mask,output_for_masks
from .learned_cache.run import nmse
METHODS=['control','mass_mean','mass_var']
class Router(Anchored):
    def __init__(self,layer,method):
        super().__init__(layer);self.method=method
        if method=='mass_var':self.alpha=nn.Parameter(torch.zeros(12))
    @torch.no_grad()
    def project(self):
        if self.method=='mass_var':self.alpha.clamp_(0.,2.)
def make_model(layer,method):return Router(layer,method)
class Events(TileEvents):
    @torch.no_grad()
    def event(self,current,q,k,v):
        e=super().event(current,q,k,v)
        full=q[0].transpose(0,1).float().reshape(12,72,64,128)
        variance=full.var(dim=2,unbiased=False)
        if self.training:
            tiles=torch.linspace(0,71,8,device=q.device).long()
            e['route_var']=variance[:,tiles].repeat_interleave(4,1)
            mass=e['z'].softmax(-1).reshape(12,8,4,361).mean(2)[...,:-1]
            e['shared_mass']=mass/mass.sum(-1,keepdim=True).clamp_min(1e-30)
        else:
            ids=torch.linspace(0,4607,32,device=q.device).long()
            e['route_var']=variance[:,ids//64]
        return e

def scores(proto,q,qvar=None,alpha=None):
    pk,_,count=proto;h,p,r,d=pk.shape
    flat=pk.flatten(1,2)
    logits=(q@flat.transpose(1,2)/math.sqrt(d)).reshape(h,q.shape[1],p,r)+count.log()[:,None]
    if alpha is not None:
        correction=(qvar@flat.square().transpose(1,2)/(2*d)).reshape_as(logits)
        logits=logits+alpha[:,None,None,None]*correction
    return logits.logsumexp(-1)

def objective(model,e,method):
    if method=='control':return previous_objective(model,e,'anchor_tile')
    z=scores(model(e['h']),e['route'][:,::4],e['route_var'][:,::4],getattr(model,'alpha',None))
    target=e['shared_mass']
    return (target*(target.clamp_min(1e-30).log()-z.log_softmax(-1))).sum(-1).mean()

@torch.no_grad()
def evaluate_event(model,e):
    proto=e['h']['mean'] if model is None else model(e['h'])
    z=scores(proto,e['route'],e['route_var'],None if model is None else getattr(model,'alpha',None))
    z=torch.cat((z,torch.zeros_like(z[...,:1])),-1)
    mask=initial_mask(z)
    assert (mask.sum(-1)==73).all() and mask[...,-1].all()
    out=output_for_masks(e['z'],e['values'],mask[:,:,None])[:,:,0]
    mse=(out-e['target']).square().sum((1,2));energy=e['target'].square().sum((1,2))
    return dict(nmse=nmse(out,e['target']).mean(-1).cpu().tolist(),sse=mse.cpu().tolist(),energy=energy.cpu().tolist(),
        max_query_nmse=nmse(out,e['target']).max(-1).values.cpu().tolist(),
        retained_mass=(e['z'].softmax(-1)*mask).sum(-1).mean(-1).cpu().tolist())
