"""Matched physical338 experiments: centered moments, page-local assignment, V-aware loss."""
import math
import torch
from torch import nn
from .response_hard import features,balanced
from .cache_grouping import reorder
from .learned_cache.data import teacher
from .learned_cache.model import output_for_masks
from .learned_cache.run import nmse
from .tile_router import Anchored
METHODS=['anchor','centered','page_local_centered','value_centered']
LAYERS=[4,9,14,19,24,29]

class Model(Anchored):
    def __init__(self,layer,method):
        super().__init__(layer);self.method=method;self.alpha=nn.Parameter(torch.zeros(12))
    @property
    def centered(self):return self.method!='anchor'
    def forward(self,h):
        k=h['k'].reshape(12,-1,64,128)
        if self.method=='page_local_centered':
            x=k-k.mean(2,keepdim=True)
            residual=4*torch.tanh(torch.einsum('hptd,hrd->hptr',x,self.weight)/4)
            labels=torch.arange(64,device=k.device)//16
            anchor=torch.nn.functional.one_hot(labels,4).to(k.dtype)*6
            a=(anchor+residual).softmax(-1);count=a.sum(2)
            pk=torch.einsum('hptr,hptd->hprd',a,k)/count[...,None].clamp_min(1e-8)
        else:
            groups=k.reshape(12,-1,4,16,128);x=groups-groups.mean(3,keepdim=True)
            a=(2*torch.tanh(torch.einsum('hprtd,hrd->hprt',x,self.weight)/2)).softmax(-1)
            pk=torch.einsum('hprt,hprtd->hprd',a,groups)
            count=torch.full(pk.shape[:3],16.,device=k.device)
        return pk,None,count
    @torch.no_grad()
    def project(self):self.alpha.clamp_(0,2)

def scores(proto,q,var,alpha,center=None):
    pk,_,count=proto;h,p,r,d=pk.shape
    z=(q@pk.flatten(1,2).transpose(1,2)/math.sqrt(d)).reshape(h,-1,p,r)+count.log()[:,None]
    shifted=pk if center is None else pk-center[:,None,None]
    z+=alpha[:,None,None,None]*(var@shifted.square().flatten(1,2).transpose(1,2)/256).reshape_as(z)
    return z.logsumexp(-1)

class Events:
    def __init__(self,layers):
        self.layers=set(layers);self.bank={l:v.cuda() for l,v in torch.load('artifacts/broad_coverage_20261030/dictionary.pt',weights_only=True).items() if l in self.layers}
        early=torch.div(torch.arange(22)*72,22,rounding_mode='floor').tolist()
        self.page_order=torch.tensor(early+[i for i in range(72) if i not in early],device='cuda');self.reset()
    def reset(self):self.chunk=None;self.history={}
    @torch.no_grad()
    def event(self,current,q,k,v):
        if current['chunk']!=self.chunk:self.history={};self.chunk=current['chunk']
        l=current['layer']
        if l not in self.history:
            kh=k[0,:-4608].transpose(0,1).float();vh=v[0,:-4608].transpose(0,1).float();orders=[]
            assert kh.shape[1]==23040
            for start in range(0,23040,4608):
                x=features(kh[:,start:start+4608],vh[:,start:start+4608],self.bank[l],'response')
                order=balanced(x,[3,3,2,2,2]);xp=reorder(x,order)
                inner=balanced(xp.reshape(-1,64,32),[2,2]).reshape(12,72,64)
                order=order.reshape(12,72,64).gather(2,inner)[:,self.page_order].flatten(1);orders.append(order+start)
            order=torch.cat(orders,1)[:,22*64:];kh=reorder(kh,order);vh=reorder(vh,order)
            self.history[l]=dict(k=kh,v=vh,center=kh.mean(1))
        h=self.history[l];full=q[0].transpose(0,1).float().reshape(12,72,64,128)
        tiles=torch.linspace(0,71,8,device=q.device).long();offsets=torch.tensor([0,21,42,63],device=q.device)
        ids=(tiles[:,None]*64+offsets).flatten();query=full.flatten(1,2)[:,ids]
        kc=k[0,-4608:].transpose(0,1).float();vc=v[0,-4608:].transpose(0,1).float()
        z,values,target=teacher(query,h['k'],h['v'],kc,vc)
        mass=z.softmax(-1).reshape(12,8,4,339).mean(2)[...,:338]
        return dict(h=h,q=query,route=full.mean(2)[:,tiles],var=full.var(2,unbiased=False)[:,tiles],kc=kc,vc=vc,
            z=z,values=values,target=target,shared_mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-30))

def objective(model,e):
    z=scores(model(e['h']),e['route'],e['var'],model.alpha,e['h']['center'] if model.centered else None)
    t=e['shared_mass'];kl=(t*(t.clamp_min(1e-30).log()-z.log_softmax(-1))).sum(-1).mean()
    if model.method!='value_centered':return kl
    # Teacher page mass and conditional V are supervision only. Online router still stores only K.
    threshold=z.topk(72,-1).values[...,-1:].detach()
    gate=torch.sigmoid((z-threshold)/.25).repeat_interleave(4,1)
    gate=torch.cat((gate,torch.ones_like(gate[...,:1])),-1)
    mass=e['z'].softmax(-1)*gate;mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-20)
    output=(mass[...,None]*e['values']).sum(-2)
    loss=(output-e['target']).square().sum(-1)/e['target'].square().sum(-1).clamp_min(1e-8)
    return .1*kl+loss.mean()

@torch.no_grad()
def evaluate(model,e):
    z=scores(model(e['h']),e['route'],e['var'],model.alpha,e['h']['center'] if model.centered else None)
    selected=z.topk(72,-1).indices
    mask=torch.zeros(12,8,339,device=z.device,dtype=torch.bool);mask.scatter_(-1,selected,True);mask[...,-1]=True
    assert (mask.sum(-1)==73).all()
    fullmask=mask.repeat_interleave(4,1)
    output=output_for_masks(e['z'],e['values'],fullmask[:,:,None])[:,:,0]
    error=nmse(output,e['target']);mass=e['z'].softmax(-1)
    metrics=dict(nmse=error.mean(-1).cpu().tolist(),max_query_nmse=error.max(-1).values.cpu().tolist(),
        retained_mass=(mass*fullmask).sum(-1).mean(-1).cpu().tolist())
    return metrics,z,mask

@torch.no_grad()
def swaps(e,z,mask,proxy=False):
    # One shared decision per tile, eight low-ranked selected x eight best unselected; includes no-op.
    drop=z.masked_fill(~mask[...,:338],float('inf')).topk(8,-1,largest=False).indices
    add=z.masked_fill(mask[...,:338],-float('inf')).topk(8,-1).indices
    truth=e['z'].softmax(-1).reshape(12,8,4,339);val=e['values'].reshape(12,8,4,339,128)
    target=e['target'].reshape(12,8,4,128)
    if proxy:
        # Fixed, data-independent four-dimensional projection. No teacher V/mass is used for selection.
        g=torch.Generator(device=z.device).manual_seed(1729)
        projection=torch.randn(12,128,4,device=z.device,generator=g)/math.sqrt(128)
        pv=e['h']['v'].reshape(12,338,64,128).mean(2)@projection
        ck=e['kc'].reshape(12,72,64,128).mean(2);cv=e['vc'].reshape(12,72,64,128).mean(2)@projection
        cz=e['route']@ck.transpose(1,2)/math.sqrt(128)+math.log(64)
        pc=cz.softmax(-1)@cv
        prob=torch.cat((z,cz.logsumexp(-1,keepdim=True)),-1).softmax(-1)[:,:,None]
        value=torch.cat((pv[:,None].expand(-1,8,-1,-1),pc[:,:,None]),2)[:,:,None]
        target_proxy=(prob[...,None]*value).sum(-2)
        delta_choice=_choose(prob,value,target_proxy,mask,drop,add)
    else:delta_choice=_choose(truth,val,target,mask,drop,add)
    chosen,improves=delta_choice
    di=chosen//8;ai=chosen%8;d=drop.gather(-1,di[...,None]);a=add.gather(-1,ai[...,None])
    corrected=mask.clone();corrected.scatter_(-1,d,~improves[...,None]);corrected.scatter_(-1,a,improves[...,None])
    assert (corrected.sum(-1)==73).all() and corrected[...,-1].all()
    out=output_for_masks(e['z'],e['values'],corrected.repeat_interleave(4,1)[:,:,None])[:,:,0]
    return dict(nmse=nmse(out,e['target']).mean(-1).cpu().tolist(),swap_fraction=improves.float().mean(-1).cpu().tolist())

def _choose(prob,value,target,mask,drop,add):
    w=prob*mask[:,:,None];den=w.sum(-1);num=(w[...,None]*value).sum(-2)
    dims=value.shape[-1];nq=value.shape[2]
    idx=lambda ids:ids[:,:,None,:].expand(-1,-1,nq,-1)
    za=prob.gather(-1,idx(drop));zb=prob.gather(-1,idx(add))
    va=value.gather(-2,idx(drop)[...,None].expand(-1,-1,-1,-1,dims))
    vb=value.gather(-2,idx(add)[...,None].expand(-1,-1,-1,-1,dims))
    newnum=num[:,:,:,None,None,:]-(za[...,None]*va)[:,:,:,:,None,:]+(zb[...,None]*vb)[:,:,:,None,:,:]
    newden=den[:,:,:,None,None]-za[:,:,:,:,None]+zb[:,:,:,None,:]
    out=newnum/newden.clamp_min(1e-20)[...,None]
    energy=target.square().sum(-1).clamp_min(1e-8)
    error=((out-target[:,:,:,None,None]).square().sum(-1)/energy[:,:,:,None,None]).mean(2)
    initial=(((num/den.clamp_min(1e-20)[...,None]-target).square().sum(-1))/energy).mean(2)
    best,choice=error.flatten(-2).min(-1)
    return choice,best<initial
