"""Matched 2x2 experiment: original mixtures vs hard-subgroup anchored weights."""
import torch
from torch import nn
from .broad_router import Events as OriginalEvents,LayerRouter,evaluate_event
from .learned_cache.data import teacher
from .learned_cache.model import predict
from .refine_pages import loss_terms as split_loss
METHODS=['random_query','random_tile','anchor_query','anchor_tile']
class Anchored(nn.Module):
    def __init__(self,layer):
        super().__init__();self.weight=nn.Parameter(torch.zeros(12,4,128))
    def forward(self,h):
        k=h['k'].reshape(12,-1,4,16,128);v=h['v'].reshape_as(k)
        # Fixed response subgroups; learn bounded normalized within-subgroup weights.
        # This is weighted aggregation, NOT a token-conserving soft partition.
        x=k-k.mean(3,keepdim=True)
        score=2*torch.tanh(torch.einsum('hbrtd,hrd->hbrt',x,self.weight)/2)
        a=score.softmax(-1)
        return (torch.einsum('hbrt,hbrtd->hbrd',a,k),torch.einsum('hbrt,hbrtd->hbrd',a,v),
                torch.full(k.shape[:3],16.,device=k.device))
def make_model(layer,method):return Anchored(layer) if method.startswith('anchor') else LayerRouter(layer)
class Events(OriginalEvents):
    training=False
    @torch.no_grad()
    def event(self,current,q,k,v):
        e=super().event(current,q,k,v)
        if not self.training:return e
        full=q[0].transpose(0,1).float()
        tiles=torch.linspace(0,71,8,device=q.device).long()
        offsets=torch.tensor([0,21,42,63],device=q.device)
        ids=(tiles[:,None]*64+offsets).flatten()
        e['q']=full[:,ids]
        e['route']=full.reshape(12,72,64,128).mean(2)[:,tiles].repeat_interleave(4,dim=1)
        e['z'],e['values'],e['target']=teacher(e['q'],e['h']['k'],e['h']['v'],e['kc'],e['vc'])
        return e
def objective(model,e,method):
    proto=model(e['h']);z,val,out=predict(e['q'],proto,e['kc'],e['vc'])
    if method.endswith('tile'):
        # Shared tile decisions supervised against four actual queries per tile.
        # Conditional V approximation remains evaluated at actual queries.
        z,_,_=predict(e['route'],proto,e['kc'],e['vc'])
    return split_loss(z,val,out,e,'split')
