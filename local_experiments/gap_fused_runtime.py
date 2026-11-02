"""Fused four-representative normalizers; preserves native reduction orders."""
import math,torch
from .gap_graph_runtime import route_features
from .high_sparse_learn import correct
from .high_sparse_gain import score
from .gap_repstats import stats
from .gap_fast_runtime import GapFastRuntime
from . import gap_graph_runtime as graph_module

def postprocess(logits,qp,base,detail,cov,vvar,kvar,w,gate,haar,training=False):
    h,t,n,p,_=logits.shape;keep=int(math.floor(p*.1+1e-6))
    logz,within=stats(logits,order=1,div=1,exp=0);prob=logz.softmax(-1);mass=prob.mean(2);feat=route_features(prob,kvar,vvar)
    weight=(prob[...,None]*within).mean(2)/mass[...,None].clamp_min(1e-12)
    coefficients=torch.einsum('htpr,ar->htpa',weight,haar)
    conditional=base.float()[:,None]+torch.einsum('htpa,hpad->htpd',coefficients,detail.float())
    # First-order diagonal projected KV-covariance proxy, not an exact page oracle.
    qbar=torch.einsum('htmp,htmd->htpd',prob,qp)/(n*mass[...,None].clamp_min(1e-12))
    residual=qbar*cov.float()[:,None]
    own=(torch.arange(t,device=logits.device)+p-t)[None].expand(h,-1)
    z=mass.detach().clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=(weighted[...,None]*conditional).sum(2)/den;allvalue=(mass[...,None]*conditional).sum(2)
    missing=allvalue-selected;diff=conditional-selected[:,:,None];energy=selected.square().sum(-1,keepdim=True).clamp_min(.01)
    align=(diff*missing[:,:,None]).sum(-1)/energy;mag=diff.square().sum(-1)/energy
    extra=torch.stack((mask.float(),(mass/den).clamp(0,2),align.clamp(-10,10)/5,mag.clamp(0,10)/5,(mass*align).clamp(-2,2),(mass*mag).clamp(0,2)),-1)
    s=score(mass,torch.cat((feat,extra),-1),w)
    contribution=mass[...,None]*residual;error=(contribution*(~mask)[...,None]).sum(2)
    # Add reduces predicted error by 2<E,e>-||e||²; dropping costs 2<E,e>+||e||².
    residual_energy=error.square().sum(-1,keepdim=True).clamp_min(1e-8)
    gain=(2*(error[:,:,None]*contribution).sum(-1)+(2*mask.float()-1)*contribution.square().sum(-1))/residual_energy
    s=s+gate.tanh()[:,None,None]*gain.clamp(-5,5)
    with torch.no_grad():
        new,_=correct(mask,s,own)
        def lut(x):return x.int().topk(keep,-1).indices.sort(-1).values[None].int()
        chosen=lut(new)
        if not training:return chosen,None,None,None
        rem=s.masked_fill(~mask,float('inf')).clone();rem.scatter_(-1,own[...,None],float('inf'));drop=rem.argmin(-1);add=s.masked_fill(mask,-float('inf')).argmax(-1)
        alt=mask.clone().scatter_(-1,drop[...,None],False).scatter_(-1,add[...,None],True);baseids=lut(mask);altids=lut(alt)
    margin=s.gather(-1,add[...,None]).squeeze(-1)-s.gather(-1,drop[...,None]).squeeze(-1)
    return chosen,baseids,altids,margin

class GapFusedRuntime(GapFastRuntime):
    def decision(self,*args,**kwargs):
        original=graph_module.postprocess
        try:
            graph_module.postprocess=postprocess
            return super().decision(*args,**kwargs)
        finally:graph_module.postprocess=original
    def finish(self):return super().finish()|dict(normalizers='fused Triton native-order logsumexp and softmax4')
