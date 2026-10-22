"""Capacity-matched, genuinely K-only and V-aware independent-pair routers."""
import math
import torch
from torch import nn
from .attention import costs,masks,block_logz


class PairMLP(nn.Module):
    def __init__(self,dims,pairs):
        super().__init__()
        self.networks=nn.ModuleList(nn.Sequential(*[
            item for i,(a,b) in enumerate(zip(dims,dims[1:]))
            for item in ([nn.Linear(a,b),nn.GELU()] if i<len(dims)-2 else [nn.Linear(a,b)])])
            for _ in range(pairs))
    def forward(self,x,pair):return self.networks[pair](x)


class KSketch(nn.Module):
    def __init__(self,pairs=36,dim=128,prototypes=8,page_size=64):
        super().__init__();self.pairs=pairs;self.dim=dim;self.prototypes=prototypes;self.page_size=page_size
        self.assignment=nn.ModuleList(nn.Linear(dim,prototypes) for _ in range(pairs))
        self.calibrator=PairMLP((6,16,1),pairs)
        self.register_buffer('feature_mean',torch.zeros(pairs,6))
        self.register_buffer('feature_std',torch.ones(pairs,6))
        self.register_buffer('feature_count',torch.zeros(pairs,dtype=torch.long))
        self.register_buffer('cal_scale',torch.ones(()))
        self.register_buffer('normalization_locked',torch.tensor(False))
        # Initial calibration is exactly zero, independently from residual heads.
        for model in self.calibrator.networks:
            nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)

    def build(self,k,pair,v=None):
        if k.ndim!=2 or k.shape[-1]!=self.dim or not len(k):raise ValueError('Bad historical K')
        b=math.ceil(len(k)/self.page_size);pad=b*self.page_size-len(k)
        kk=torch.nn.functional.pad(k.float(),(0,0,0,pad)).reshape(b,self.page_size,self.dim)
        valid=(torch.arange(b*self.page_size,device=k.device).reshape(b,self.page_size)<len(k))
        a=self.assignment[pair](kk).softmax(-1)*valid[...,None]
        count=a.sum(1);denom=count.clamp_min(1e-12)[...,None]
        mean=torch.einsum('btp,btd->bpd',a,kk)/denom
        var=(torch.einsum('btp,btd->bpd',a,kk.square())/denom-mean.square()).clamp_min(0)
        state=dict(count=count,key=mean,variance=var,costs=costs(len(k),self.page_size,k.device),
                   history=len(k),pair=pair)
        # No V access at all in the K branch, including during cache construction.
        if v is not None:
            if v.shape!=k.shape:raise ValueError('K/V shape mismatch')
            vv=torch.nn.functional.pad(v.float(),(0,0,0,pad)).reshape_as(kk)
            state['value']=torch.einsum('btp,btd->bpd',a,vv)/denom
        return state

    def raw(self,q,state):
        logits=torch.einsum('qd,bpd->qbp',q.float(),state['key'])/math.sqrt(self.dim)
        logits=logits+state['count'].clamp_min(1e-30).log()[None]
        valid=state['count']>1e-8
        logits=logits.masked_fill(~valid[None],-torch.inf)
        weights=logits.softmax(-1);raw=logits.logsumexp(-1)
        entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)
        qnorm=q.float().norm(dim=-1).clamp_min(1e-12)
        direction=q.float()/qnorm[:,None]
        prototype_count=valid.sum(-1).clamp_min(1)[None]
        finite_logits=torch.where(valid[None],logits,torch.zeros_like(logits))
        logit_mean=finite_logits.sum(-1)/prototype_count
        logit_var=torch.where(valid[None],(finite_logits-logit_mean[...,None]).square(),torch.zeros_like(logits)).sum(-1)/prototype_count
        kvar=(torch.einsum('qd,bpd->qbp',direction.square(),state['variance'])*valid[None]).sum(-1)/prototype_count
        features=torch.stack((qnorm.log()[:,None].expand_as(raw),
            state['costs'].log()[None].expand_as(raw),logit_var.clamp_min(1e-12).sqrt(),
            logits.amax(-1)-logit_mean,entropy,kvar),-1)
        return raw,weights,features,entropy

    def estimate(self,q,state,kc,vc=None,kind='K'):
        raw,w,features,entropy=self.raw(q,state);pair=state['pair']
        norm=(features-self.feature_mean[pair])/self.feature_std[pair].clamp_min(1e-3)
        logz=raw+self.calibrator(norm,pair).squeeze(-1)
        current=q.float()@kc.float().T/math.sqrt(self.dim)
        lc=current.logsumexp(-1);wc=current.softmax(-1)
        mu=torch.einsum('qbp,bpd->qbd',w,state['key' if kind=='K' else 'value'])
        # Both paths are active, equally dimensioned content representations.
        oc=wc@(kc if kind=='K' else vc).float()
        total=torch.cat((logz.double(),lc.double()[:,None]),-1)
        mass=total.softmax(-1);mh=mass[:,:-1];mc=mass[:,-1]
        dense=(mh[...,None]*mu.double()).sum(1)+mc[:,None]*oc.double()
        residual=mh[...,None]*(mu.double()-dense[:,None])
        return dict(raw_logz=raw,logz=logz,current_logz=lc,mu=mu,dense=dense.float(),
                    mass=mh.float(),current_mass=mc.float(),residual=residual.float(),
                    entropy=entropy,cal_features=features)

    def calibration_loss(self,q,kh,kc,state,estimate):
        # Detached CURRENT Student QK labels, not frozen teacher historical labels.
        with torch.no_grad():
            truth=block_logz(q.detach(),kh.detach(),self.page_size)
            lc=(q.detach().float()@kc.detach().float().T/math.sqrt(self.dim)).logsumexp(-1)
        loss=torch.nn.functional.huber_loss(estimate['logz']-lc[:,None],truth-lc[:,None])
        return loss/self.cal_scale.clamp_min(1e-8)


class InformationRouter(nn.Module):
    def __init__(self,kind,pairs=36,dim=128,page_size=64,prototypes=8):
        super().__init__()
        if kind not in ('K','V'):raise ValueError(kind)
        self.kind=kind;self.sketch=KSketch(pairs,dim,prototypes,page_size)
        self.content=nn.ModuleList(nn.Linear(dim,8) for _ in range(pairs))
        self.norm=nn.ModuleList(nn.LayerNorm(8) for _ in range(pairs))
        self.head=PairMLP((17,32,1),pairs)
        self.scale=nn.Parameter(torch.full((pairs,),.1))
        self.reset_residual()

    def reset_residual(self):
        for h in self.head.networks:
            nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
        with torch.no_grad():self.scale.fill_(.1)

    def build(self,k,v,pair,detach_inputs=True):
        if detach_inputs:k=k.detach();v=v.detach()
        return self.sketch.build(k,pair,v if self.kind=='V' else None)

    def score(self,q,kc,vc,state,keep=.2,step=0,detach_inputs=True,mass_only=False):
        if detach_inputs:q=q.detach();kc=kc.detach();vc=vc.detach()
        e=self.sketch.estimate(q,state,kc,vc if self.kind=='V' else None,self.kind)
        pair=state['pair'];r=e['residual'];n=r.norm(dim=-1).clamp_min(1e-12)
        projection=self.norm[pair](self.content[pair](r))
        base=e['logz'];const=lambda value:torch.full_like(base,float(value))
        scalar=torch.stack((base-e['current_logz'][:,None],state['costs'].log()[None].expand_as(base),
            const(keep),q.float().norm(dim=-1).clamp_min(1e-12).log()[:,None].expand_as(base),
            const(step),const(math.log(state['history'])),e['current_mass'][:,None].expand_as(base),e['entropy']),-1)
        feature=torch.cat((scalar,projection,n.log()[...,None]),-1)
        delta=self.scale[pair]*self.head(feature,pair).squeeze(-1)
        scores=base if mass_only else base+delta
        return dict(scores=scores,delta=delta,estimate=e,features=feature)

    def forward(self,q,kc,vc,state,keep=.2,step=0,detach_inputs=True,mass_only=False):
        result=self.score(q,kc,vc,state,keep,step,detach_inputs,mass_only)
        mask,hard,soft=masks(result['scores'],state['costs'],keep)
        return dict(result,mask=mask,hard=hard,soft=soft)
