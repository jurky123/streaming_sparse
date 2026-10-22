"""Residual scoring and one set correction; teacher geometry is excluded by API."""
import math
import torch
from torch import nn
from ..v03.model import PairMLP
from ..v05.primitives import budget_mask,select
from .sketch import Sketch
from .exact import singleton,set_gain
from .common import tensor_digest


class BaseScorer(nn.Module):
    def __init__(self):
        super().__init__();self.calibration=PairMLP((8,32,1),36,1);self.gamma=nn.Parameter(torch.ones(36))
        with torch.no_grad():
            self.calibration.linears[-1].base.weight.mul_(.01)
            self.calibration.linears[-1].base.bias.mul_(.01)
    def forward(self,q,g,costs,pair,rho,history_only=False):
        logerr=singleton(g).clamp_min(1e-12).log();logm=g['mass'].clamp_min(1e-30).log().clamp_min(-40)
        qn=q.float().norm(dim=-1).clamp_min(1e-12).log()[:,None].expand_as(logm)
        c=costs.log()[None].expand_as(logm);budget=torch.full_like(logm,rho)
        cur=torch.zeros_like(logm) if history_only else (g['current_logz']-g['total_logz'])[:,None].expand_as(logm).clamp_min(-40)
        feature=torch.stack((logm,g['d'].norm(dim=-1).clamp_min(1e-12).log(),c,budget,cur,qn,
                             logerr,g['mu'].norm(dim=-1).clamp_min(1e-12).log()),-1)
        scores=logerr-self.gamma[pair]*c+self.calibration(feature,pair).squeeze(-1)
        return scores,feature


class Correction(nn.Module):
    def __init__(self):
        super().__init__();self.head=PairMLP((4,16,1),36,1);self.alpha_raw=nn.Parameter(torch.full((36,),math.log(math.expm1(.1))))
        with torch.no_grad():
            self.head.linears[-1].base.weight.mul_(.01);self.head.linears[-1].base.bias.mul_(.01)
    def forward(self,first,g,costs,pair,rho):
        logm=g['mass'].clamp_min(1e-30).log().clamp_min(-40)
        inputs=torch.stack((first.clamp(-40,40),logm,costs.log()[None].expand_as(first),torch.full_like(first,rho)),-1)
        alpha=torch.nn.functional.softplus(self.alpha_raw[pair])
        return alpha*first.tanh()+self.head(inputs,pair).squeeze(-1),alpha


class Router(nn.Module):
    def __init__(self,sketch_config=None,arms=()):
        super().__init__();self.sketch=Sketch(**(sketch_config or {}));self.base=BaseScorer();self.arms=tuple(arms)
        if not set(arms)<={'R3-local','R4'}:raise ValueError(arms)
        self.corrections=nn.ModuleDict({name:Correction() for name in arms})
        if len(arms)>1:
            first=self.corrections[arms[0]].state_dict()
            for name in arms[1:]:self.corrections[name].load_state_dict(first)
        self.base_frozen=False;self.sketch_frozen=False
    def base_digest(self):return tensor_digest([(n,v) for n,v in self.state_dict().items() if not n.startswith('corrections.')])
    def freeze_base(self):
        self.sketch.requires_grad_(False);self.base.requires_grad_(False);self.base_frozen=True;self.sketch_frozen=True
    def checkpoint(self):return dict(sketch_config=self.sketch.config(),arms=self.arms,router=self.state_dict(),base_digest=self.base_digest())
    def route(self,q,state,current,keep=.2,history_only=False,shuffle=False):
        pair=state['pair'];costs=state['costs'];B=len(costs);budget=math.floor(keep*state['history_tokens'])
        # No target, true dense, exact historical statistics, or raw history KV can enter this path.
        g=self.sketch.estimate(q,state,current,history_only=history_only)
        r,features=self.base(q,g,costs,pair,keep,history_only)
        hard=select(r,costs,budget)
        # Pair-specific costs fixed: use original analytic implicit budget derivative.
        identity=torch.diag(costs)
        initial,_,soft=budget_mask(r,identity,identity,budget,.2)
        m0=hard.detach() if self.base_frozen else initial
        out={'R2' if history_only else 'R3':dict(scores=r,mask=initial,hard_mask=hard,soft_mask=soft,initial_mask=hard,
                                               initial_scores=r,geometry=g,features=features)}
        if not history_only:
            mass_scores=g['mass'].clamp_min(1e-30).log()
            mm,mh,ms=budget_mask(mass_scores,identity,identity,budget,.2)
            out['R1']=dict(scores=mass_scores,mask=mm,hard_mask=mh,soft_mask=ms,initial_mask=mh,geometry=g)
        for name,module in self.corrections.items():
            if name=='R3-local':first=singleton(g).clamp_min(1e-12).log();condition=None
            else:
                conditioned=m0.roll(1,0) if shuffle else m0
                condition=set_gain(g,conditioned);first=condition['u']
            delta,alpha=module(first,g,costs,pair,keep)
            scores=r+delta;mask,hh,ss=budget_mask(scores,identity,identity,budget,.2)
            out[name]=dict(scores=scores,mask=mask,hard_mask=hh,soft_mask=ss,initial_mask=hard,initial_scores=r,
                           geometry=g,delta=delta,alpha=alpha,set_condition=condition)
        return out


def load(path,device='cpu'):
    saved=torch.load(path,map_location=device,weights_only=True)
    if saved['kind']=='sketch':
        model=Sketch(**saved['config']).to(device);model.load_state_dict(saved['state'])
    else:
        model=Router(saved['sketch_config'],saved['arms']).to(device);model.load_state_dict(saved['router'])
        if model.base_digest()!=saved['base_digest']:raise ValueError('Base checksum mismatch')
    return model,saved
