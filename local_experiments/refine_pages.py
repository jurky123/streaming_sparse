"""Response initialization, explicit mass/value/rank objectives, bounded calibration."""
import math
import json
from pathlib import Path
import torch
from torch import nn
from .page_local import PageMixture
from .learned_pool import Bank,DICTIONARY
from .learned_cache.data import records
from .learned_cache.model import predict
from .learned_cache.io import atomic,digest

ROOT=Path('artifacts/refine_pages_20261030')


@torch.no_grad()
def initialize():
    """Train-only response PCA -> four response bands, retaining affine K kernel."""
    bank=Bank(4,max_history=1,max_events=1);torch.manual_seed(0);model=PageMixture().cuda()
    manifest={str(DICTIONARY):digest(DICTIONARY)}
    for layer in [4,14,24]:
        paths=[p for p in records('train') if f'l{layer:02d}' in p.name and p.stem.endswith('s0')]
        keys=[]
        for path in paths:
            hp=path.with_name(path.name[:-6]+'_history.pt');manifest[str(hp)]=digest(hp)
            keys.append(torch.load(hp,weights_only=True)['k'].cuda().float())
        k=torch.cat(keys,1);q=bank.dictionary[layer]
        response=k@q.transpose(1,2)/math.sqrt(128);mean=response.mean(1,keepdim=True);x=response-mean
        covariance=x.transpose(1,2)@x/x.shape[1]
        direction=torch.zeros_like(covariance[:,:,0]).scatter_(1,covariance.diagonal(dim1=-2,dim2=-1).argmax(-1)[:,None],1.)
        for _ in range(16):
            direction=(covariance@direction[:,:,None]).squeeze(-1);direction/=direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        coeff=(direction[:,None]@q).squeeze(1)/math.sqrt(128)
        projection=(k*coeff[:,None]).sum(-1);center=projection.mean(-1);scale=projection.std(-1,unbiased=False).clamp_min(1e-5)
        bands=torch.tensor([-1.5,-.5,.5,1.5],device='cuda')
        # -(standardized response-band)^2: the common quadratic term cancels.
        model.weight[layer,[0,4,8]]=2*bands[None,:,None]*coeff[:,None]/scale[:,None,None]
        model.bias[layer,[0,4,8]]=-2*bands[None]*center[:,None]/scale[:,None]-bands.square()[None]
    atomic(ROOT/'response_initial.pt',model.state_dict(),True)
    atomic(ROOT/'initialization_manifest.json',dict(inputs=manifest,recipe='32-response PCA,16 power iterations, bands -1.5,-.5,.5,1.5, training history only'))


class Calibrated(nn.Module):
    def __init__(self):
        super().__init__();self.base=PageMixture()
        choice=json.loads((ROOT/'calibration_base.json').read_text())
        assert digest(choice['path'])==choice['sha256']
        self.base.load_state_dict(torch.load(choice['path'],weights_only=False,map_location='cpu')['model']);self.base.requires_grad_(False)
        self.net=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        dictionary=torch.load(DICTIONARY,weights_only=True)
        for layer,q in dictionary.items():self.register_buffer(f'basis_{layer}',torch.linalg.qr(q[:,:8].transpose(1,2).float()).Q)

    def forward(self,h):return self.base(h)

    def correction(self,q,h,proto):
        pk,_,count=proto;basis=getattr(self,f'basis_{h["layer"]}')
        if 'variance8' not in h:
            with torch.no_grad():
                projected=(h['k']@basis).reshape(3,-1,64,8)
                h['variance8']=(projected-projected.mean(2,keepdim=True)).square().mean(2)
        scores=torch.einsum('hqd,hprd->hqpr',q,pk)/math.sqrt(128)
        logit=scores+count.clamp_min(1e-20).log()[:,None];prob=logit.softmax(-1);z=logit.logsumexp(-1)
        center=(prob*scores).sum(-1);spread=(prob*(scores-center[...,None]).square()).sum(-1)
        mass=count/64;entropy=-(mass*mass.clamp_min(1e-20).log()).sum(-1)/math.log(4)
        qproj=q@basis;variance=torch.einsum('hqr,hpr->hqp',qproj.square(),h['variance8'])/128
        expand=lambda t:t[:,None].expand_as(z)
        feat=torch.stack((z.clamp(-40,40)/20,spread.log1p()/5,(scores.max(-1).values-scores.min(-1).values).log1p()/5,
            q.norm(dim=-1).log1p()[:,:,None].expand_as(z)/5,expand(entropy),expand(mass.max(-1).values),
            expand(pk.square().mean((-1,-2)).sqrt().log1p())/5,variance.log1p()/5),-1)
        return .5*torch.tanh(self.net(feat.clamp(-10,10)).squeeze(-1))


def make_model(rep):
    if rep=='calibration':return Calibrated()
    model=PageMixture()
    if rep=='response':model.load_state_dict(torch.load(ROOT/'response_initial.pt',weights_only=True,map_location='cpu'))
    return model


def prediction(model,e,rep,route=True):
    q=e['route' if route else 'q'];proto=model(e['h']);z,val,out=predict(q,proto,e['kc'],e['vc'])
    if rep=='calibration':
        z=torch.cat((z[...,:-1]+model.correction(q,e['h'],proto),z[...,-1:]),-1)
        out=(z.softmax(-1)[...,None]*val).sum(2)
    return z,val,out


def loss_terms(z,val,out,e,objective):
    from .learned_cache.run import nmse
    if objective=='reconstruction':
        mass=e['z'].softmax(-1)
        page=((val-e['values']).square().mean(-1)*mass).sum(-1)/e['target'].square().mean(-1).clamp_min(1e-5)
        return nmse(out,e['target']).mean()+.01*((z-e['z']).square()*mass).sum(-1).mean()+.1*page.mean()
    mass=e['z'][...,:-1].softmax(-1)
    quality=((z[...,:-1]-e['z'][...,:-1]).square()*mass).sum(-1).mean()
    page=((val[:,:,:-1]-e['values'][:,:,:-1]).square().mean(-1)*mass).sum(-1)/e['target'].square().mean(-1).clamp_min(1e-5)
    # Rank 8 selected boundary pages against 8 omitted boundary pages.
    order=z[...,:-1].detach().argsort(-1,descending=True);left=order[...,64:72];right=order[...,72:80]
    pd=z.gather(-1,left)[...,None]-z.gather(-1,right)[:,:,None]
    td=e['z'].gather(-1,left)[...,None]-e['z'].gather(-1,right)[:,:,None]
    rank=(torch.nn.functional.softplus(-pd*td.sign())*td.abs().clamp(max=1)).mean()
    return .01*quality+.1*rank+(.1*page.mean() if objective=='split' else 0)
