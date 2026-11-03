"""Bounded page-streamed quality references. No production latency claim.

Shapes: raw H,P,64,D; queries H,Q,D; summary H,P,R,D.
Only query/probe slices are evaluated here; no H,T,M,P,D residual tensor.
"""
from dataclasses import dataclass
import math
import torch

HAAR = ((1,1,-1,-1),(1,-1,1,-1),(1,-1,-1,1))

@dataclass(frozen=True)
class Config:
    history_pages: int = 338
    current_pages: int = 72
    probes: int = 16
    density: float = .10
    max_fraction: float = .25
    id_capacity: int = 128
    def __post_init__(self):
        assert self.history_pages >= 72 and self.current_pages == 72
        assert 1 <= self.probes <= 64 and 0 < self.density <= self.max_fraction <= 1
        assert self.id_capacity >= math.ceil(self.max_fraction*(self.history_pages+self.current_pages))
    @property
    def drop(self): return (self.current_pages-self.history_pages%self.current_pages)%self.current_pages
    def page_order(self, device):
        early=[i*self.current_pages//self.drop for i in range(self.drop)] if self.drop else []
        return torch.tensor(early+[i for i in range(self.current_pages) if i not in early],device=device)

@dataclass
class Summary:
    keys: torch.Tensor
    mean: torch.Tensor
    detail: torch.Tensor
    stats: torch.Tensor
    def bytes(self): return sum(t.numel()*t.element_size() for t in (self.keys,self.mean,self.detail,self.stats))
    def slice(self, a, b): return Summary(*(getattr(self,n)[:,a:b] for n in ('keys','mean','detail','stats')))


def haar(x): return x.new_tensor(HAAR, dtype=torch.float32)
def deployed(x): return x.to(torch.bfloat16).float()

class SummaryOps:
    @staticmethod
    def build(model, k, v, cheap=False):
        # Raw KV is already rounded by the cache; statistics use precisely that state.
        k=deployed(k);v=deployed(v);h,p,t,d=k.shape
        assert (t,d)==(64,128)
        kg=k.reshape(h,p,4,16,d);vg=v.reshape_as(kg)
        if cheap:
            f=torch.einsum('hptd,hrd->hptr',k-k.mean(2,keepdim=True),model.bank)/math.sqrt(d)
            f=f/f.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        else: f=model.group_features(k,v)
        f=f.reshape(h,p,4,16,32)
        z=torch.einsum('hprtd,hrd->hprt',kg-kg.mean(3,keepdim=True),model.pool_k)
        z=z+torch.einsum('hprtd,hrd->hprt',vg-vg.mean(3,keepdim=True),model.pool_v)
        z=z+torch.einsum('hprtf,hrf->hprt',f,model.pool_feature)
        a=(2*torch.tanh((z-z.mean(-1,keepdim=True))/2)).softmax(-1)
        pk=(a[...,None]*kg).sum(3);pv=(a[...,None]*vg).sum(3)
        pk=deployed(pk-pk.mean(2,keepdim=True)+k.mean(2,keepdim=True))
        mu=deployed(v.mean(2));basis=deployed(model.basis)
        code=torch.einsum('hprd,hdk->hprk',pv,basis)
        detail=deployed(torch.einsum('ar,hprk->hpak',haar(k),code)/4)
        reconstructed=mu[:,:,None]+torch.einsum('ar,hpak,hdk->hprd',haar(k),detail,basis)
        # Slots 0:8 D; 8 bK; 9 bV; 10 sV; slot11 stores page scalars:
        # rep0 bV0, rep1 Bp, reps2/3 reserved=0. No hidden auxiliary cache.
        km=kg.mean(3);vm=vg.mean(3)
        stats=k.new_zeros(h,p,4,12)
        stats[...,:8]=(kg-km[:,:,:,None]).square().mean(3).reshape(h,p,4,8,16).mean(-1)
        stats[...,8]=(km-pk).norm(dim=-1)
        stats[...,9]=(vm-reconstructed).norm(dim=-1)
        stats[...,10]=(vg-vm[:,:,:,None]).square().sum(-1).mean(3).sqrt()
        stats[:,:,0,11]=(v.mean(2)-mu).norm(dim=-1)
        stats[:,:,1,11]=v.norm(dim=-1).amax(2)
        return Summary(*(x.to(torch.bfloat16).contiguous() for x in (pk,mu,detail,stats)))

class RouteOps:
    @staticmethod
    def features(q, summary, valid=None, block=16):
        """Two QK passes; output H,Q,P,4 scalar features, no conditional V vectors.
        Training/calibration reference retains these 4 features. Inference can
        contract theta blockwise; a fused production implementation is deferred.
        """
        h,n,d=q.shape;p=summary.keys.shape[1];q=deployed(q)
        if valid is None: valid=torch.ones(h,p,device=q.device,dtype=torch.bool)
        assert valid.any(-1).all()
        logz=q.new_full((h,n),-torch.inf)
        for a in range(0,p,block):
            keys=summary.keys[:,a:a+block].float()
            z=torch.einsum('hqd,hprd->hqpr',q,keys)/math.sqrt(d)
            z=z.masked_fill(~valid[:,None,a:a+block,None],-torch.inf)
            logz=torch.logaddexp(logz,z.flatten(2).logsumexp(-1))
        q2=q.square().reshape(h,n,8,16).sum(-1)/d;qn=q.norm(dim=-1)/math.sqrt(d)
        bp=summary.stats[:,:,1,11].float();bmax=bp.masked_fill(~valid,0).amax(-1)
        features=[]
        for a in range(0,p,block):
            s=summary.slice(a,a+block);st=s.stats.float()
            z=torch.einsum('hqd,hprd->hqpr',q,s.keys.float())/math.sqrt(d)
            z=z.masked_fill(~valid[:,None,a:a+block,None],-torch.inf)
            prob=(z-logz[:,:,None,None]).exp()
            alpha=prob.sum(-1)
            u=torch.einsum('hqg,hprg->hqpr',q2,st[...,:8]).clamp_min(0)
            beta=qn[:,:,None,None]*st[:,None,:,:,8]
            c=bp[:,None,a:a+block]+bmax[:,None,None]
            x0=alpha*st[:,None,:,0,11]+((prob-alpha[...,None]/4).abs()*st[:,None,:,:,9]).sum(-1)
            x1=(prob*u.sqrt()*st[:,None,:,:,10]).sum(-1)
            x2=c*(prob*beta).sum(-1)
            x3=c*(prob*(u+beta.square())).sum(-1)
            features.append(torch.stack((x0,x1,x2,x3),-1))
        return torch.cat(features,2),logz

    @staticmethod
    def select(scores, own, config, threshold=None, valid=None):
        """Device counts and fixed capacity. Padding -1 MUST NOT be read.
        own H,T (one protected current page); scores H,T,P.
        """
        h,t,p=scores.shape
        if valid is None:valid=torch.ones_like(scores,dtype=torch.bool)
        valid=valid.expand_as(scores)
        assert valid.gather(-1,own[...,None]).all()
        cap=min(config.id_capacity,math.ceil(config.max_fraction*p))
        ranked=scores.masked_fill(~valid,-torch.inf).scatter(-1,own[...,None],torch.inf)
        vals,order=ranked.topk(min(config.id_capacity,p),dim=-1)
        if threshold is None:
            count=torch.full((h,t),max(1,math.floor(p*config.density)),device=scores.device,dtype=torch.long)
            count=torch.minimum(count,valid.sum(-1));overflow=torch.zeros_like(count)
        else:
            eligible=(scores>threshold)&valid
            eligible=eligible.scatter(-1,own[...,None],True)
            wanted=eligible.sum(-1);overflow=(wanted-cap).clamp_min(0);count=wanted.clamp(1,cap)
        active=torch.arange(order.shape[-1],device=scores.device)<count[...,None]
        ids=order.masked_fill(~active,-1)
        if ids.shape[-1]<config.id_capacity:
            ids=torch.nn.functional.pad(ids,(0,config.id_capacity-ids.shape[-1]),value=-1)
        mask=torch.zeros_like(scores,dtype=torch.bool).scatter(-1,order,active)
        residual=(scores.masked_fill(mask|~valid,0)).sum(-1)
        return dict(ids=ids.int(),counts=count.int(),mask=mask,overflow=overflow,residual_risk=residual)

class AttentionOps:
    @staticmethod
    def pages(q,k,v,s,basis):
        """Exact/summary page logZ and normalized numerator; common log domain."""
        q=deployed(q);basis=deployed(basis);d=q.shape[-1]
        z=torch.einsum('hqd,hptd->hqpt',q,k.float())/math.sqrt(d)
        lz=z.logsumexp(-1);prob=z.softmax(-1)
        exact=torch.einsum('hqpt,hptd->hqpd',prob,v.float())
        zs=torch.einsum('hqd,hprd->hqpr',q,s.keys.float())/math.sqrt(d)
        ls=zs.logsumexp(-1)+math.log(16);w=zs.softmax(-1)
        contrast=torch.einsum('hqpr,ar->hqpa',w,haar(q))
        # Aggregate representative weights before V multiplication: zero-sum intact.
        code=torch.einsum('hqpa,hpak->hqpk',contrast,s.detail.float())
        approx=s.mean.float()[:,None]+torch.einsum('hqpk,hdk->hqpd',code,basis)
        return exact,lz,approx,ls

    @staticmethod
    def evaluate(q,k,v,summary,basis,mask,block=8):
        """Page-streamed mathematical reference. Mask H,Q,P is already expanded.
        Computes exact for all pages FOR VALIDATION ONLY, not sparse speed path.
        """
        h,n,d=q.shape;p=k.shape[1];out=q.new_zeros(h,n,d,dtype=torch.float32)
        logz=out.new_full((h,n),-torch.inf)
        for a in range(0,p,block):
            exact,lz,approx,ls=AttentionOps.pages(q,k[:,a:a+block],v[:,a:a+block],summary.slice(a,a+block),basis)
            select=mask[:,:,a:a+block];z=torch.where(select,lz,ls)
            values=torch.where(select[...,None],exact,approx)
            chunkz=z.logsumexp(-1);newz=torch.logaddexp(logz,chunkz)
            out=out*(logz-newz).exp()[...,None]+torch.einsum('hqp,hqpd->hqd',(z-newz[...,None]).exp(),values)
            logz=newz
        return out,logz

    @staticmethod
    def residual_labels(q,k,v,s,basis,teacher,teacher_logz,block=8):
        """Normalized actual e_p labels, no raw residual vectors retained."""
        labels=[]
        for a in range(0,k.shape[1],block):
            exact,lz,approx,ls=AttentionOps.pages(q,k[:,a:a+block],v[:,a:a+block],s.slice(a,a+block),basis)
            # Offline labels can greatly exceed the final normalized attention error.
            # FP64 ratios and scaled norms avoid FP32 exp/square overflow; the
            # deployed attention path and its FP32 log-normalizers are unchanged.
            ep=(lz.double()-teacher_logz.double()[:,:,None]).exp()[...,None]*(exact.double()-teacher.double()[:,:,None])
            ep=ep-(ls.double()-teacher_logz.double()[:,:,None]).exp()[...,None]*(approx.double()-teacher.double()[:,:,None])
            magnitude=ep.abs().amax(-1)
            norm=magnitude*(ep/magnitude.clamp_min(torch.finfo(torch.float64).tiny)[...,None]).square().sum(-1).sqrt()
            assert torch.isfinite(norm).all(), 'Residual labels exceed FP64 range; do not silently clip'
            labels.append(norm)
        return torch.cat(labels,-1)
