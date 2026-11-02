"""Deploy selected pilot models only in audited layers; same raw/selected KV caps."""
import math
from pathlib import Path
import torch
from .improve_runtime import FastRuntime
from .improve_models import Model,_choose
from .response_hard import features,balanced
from .cache_grouping import reorder
from .rollout_kernels import route_scores

class LearnedRuntime(FastRuntime):
    def __init__(self,engine,method,checkpoint='artifacts/rollout_improve_20261030/pilot/best.pt',proxy=False,audit_scores=False):
        super().__init__(engine,'route_fast');self.models={};self.centers={};self.logcounts={};self.vectors={};self.projectors={};self.proxy=proxy;self.audit_scores=audit_scores
        saved=torch.load(checkpoint,weights_only=False,map_location='cpu')['models'][method]
        for l,state in saved.items():
            model=Model(l,method).cuda().eval().requires_grad_(False);model.load_state_dict(state);self.models[l]=model
            self.weights[l]=model.weight;self.alpha[l]=model.alpha
            if model.centered:self.centers[l]=torch.zeros(12,128,device='cuda')
            if method=='page_local_centered':self.logcounts[l]=torch.empty(12,self.cap,4,device='cuda')
            if proxy:
                g=torch.Generator(device='cuda').manual_seed(1729)
                self.projectors[l]=torch.randn(12,128,4,device='cuda',generator=g)/math.sqrt(128)
                self.vectors[l]=torch.empty(12,self.cap,4,device='cuda')
        self.method='pilot_'+method+('_swap' if proxy else '')
        self.persistent_bytes+=sum(x.numel()*x.element_size() for group in [self.centers,self.logcounts,self.vectors,self.projectors] for x in group.values())
        assert self.persistent_bytes<=self.budget,(self.persistent_bytes,self.budget)
    def reset(self):
        super().reset()
        for center in self.centers.values():center.zero_()
    def commit(self,l,k,v,cache):
        if l not in self.models:return super().commit(l,k,v,cache)
        kh=k[0].transpose(0,1).float();vh=v[0].transpose(0,1)
        x=features(kh,vh,self.bank[l],'response');order=balanced(x,[3,3,2,2,2]);xp=reorder(x,order)
        inner=balanced(xp.reshape(-1,64,32),[2,2]).reshape(12,72,64)
        order=order.reshape(12,72,64).gather(2,inner)[:,self.page_order].flatten(1)
        kr=reorder(kh,order);vr=reorder(vh,order)
        pk,_,count=self.models[l](dict(k=kr))
        slots=(torch.arange(72,device=k.device)+self.write[l])%self.cap
        if l in self.centers:
            evict=max(0,self.n[l]+72-self.cap)
            total=self.centers[l]*(self.n[l]*64)
            if evict:
                ids=(slots[-evict:,None]*64+self.offsets).flatten()
                total-=cache['k'][0].index_select(0,ids).float().sum(0)
            total+=kh.sum(1)
            self.centers[l].copy_(total/(min(self.cap,self.n[l]+72)*64))
        ids=(slots[:,None]*64+self.offsets).flatten()
        cache['k'][0].index_copy_(0,ids,kr.transpose(0,1).to(torch.bfloat16));cache['v'][0].index_copy_(0,ids,vr.transpose(0,1))
        self.pk[l].index_copy_(1,slots,pk)
        if l in self.logcounts:self.logcounts[l].index_copy_(1,slots,count.log()-math.log(16))
        if self.proxy:self.vectors[l].index_copy_(1,slots,vr.float().reshape(12,72,64,128).mean(2)@self.projectors[l])
        self.write[l]=(self.write[l]+72)%self.cap;self.n[l]=min(self.cap,self.n[l]+72);self.commits[l]+=1
    def route(self,l,q,p):
        if l not in self.models:return super().route(l,q,p)
        from .improve_learned_kernels import route
        z=route(q,self.pk[l][:,:p],self.alpha[l],self.centers.get(l),None if l not in self.logcounts else self.logcounts[l][:,:p])
        if self.audit_scores:
            full=q[0].transpose(0,1).float().reshape(12,72,64,128);mean=full.mean(2);var=full.var(2,unbiased=False);pk=self.pk[l][:,:p]
            raw=(mean@pk.flatten(1,2).transpose(1,2)/math.sqrt(128)).reshape(12,72,p,4)
            shifted=pk if l not in self.centers else pk-self.centers[l][:,None,None]
            raw+=self.alpha[l][:,None,None,None]*(var@shifted.square().flatten(1,2).transpose(1,2)/256).reshape_as(raw)
            if l in self.logcounts:raw+=self.logcounts[l][:,None,:p]
            torch.testing.assert_close(z,raw.logsumexp(-1),atol=8e-4,rtol=5e-4)
        chosen=z.topk(72,-1,sorted=False).indices
        if self.proxy:
            assert p==338 or p in [144,216,288]
            # Student-only, same single swap for all 64 queries in a tile.
            ck=self.engine.cache[l]['k'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,64,128).mean(2)
            cv=self.engine.cache[l]['v'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,64,128).mean(2)@self.projectors[l]
            qm=q[0].transpose(0,1).float().reshape(12,72,64,128).mean(2)
            cz=qm@ck.transpose(1,2)/math.sqrt(128)+math.log(64);pc=cz.softmax(-1)@cv
            prob=torch.cat((z+math.log(16),cz.logsumexp(-1,keepdim=True)),-1).softmax(-1)[:,:,None]
            value=torch.cat((self.vectors[l][:,None,:p].expand(-1,72,-1,-1),pc[:,:,None]),2)[:,:,None]
            target=(prob[...,None]*value).sum(-2)
            mask=torch.zeros(12,72,p+1,device=q.device,dtype=torch.bool);mask.scatter_(-1,chosen,True);mask[...,-1]=True
            drop=z.masked_fill(~mask[...,:p],float('inf')).topk(8,-1,largest=False).indices
            add=z.masked_fill(mask[...,:p],-float('inf')).topk(8,-1).indices
            choice,improve=_choose(prob,value,target,mask,drop,add)
            d=drop.gather(-1,(choice//8)[...,None]);a=add.gather(-1,(choice%8)[...,None])
            mask.scatter_(-1,d,~improve[...,None]);mask.scatter_(-1,a,improve[...,None])
            chosen=mask[...,:p].to(torch.int32).topk(72,-1).indices
        ids=chosen.sort(-1).values.to(torch.int32)
        current=torch.arange(p,p+72,device=q.device,dtype=torch.int32).expand(12,72,72)
        return torch.cat((ids,current),-1)[None]
