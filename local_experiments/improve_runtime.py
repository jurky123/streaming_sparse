"""Isolated engineering variants and late-rollout correctness checks."""
import math
import torch
from .rollout_runtime import Runtime
from .rollout_kernels import route_scores,aggregate
from .response_hard import features,balanced
from .cache_grouping import reorder

class FastRuntime(Runtime):
    def __init__(self,engine,variant='route_fast',check_routes=False):
        super().__init__(engine,'mass_var');self.variant=variant;self.check_routes=check_routes;self.route_differences=0
    def route(self,l,q,p):
        z=route_scores(q,self.pk[l][:,:p],self.alpha.get(l))
        ids=z.topk(72,-1,sorted=False).indices.sort(-1).values.to(torch.int32)
        current=torch.arange(p,p+72,device=q.device,dtype=torch.int32).expand(12,72,72)
        lut=torch.cat((ids,current),-1)[None]
        if self.check_routes:
            ref=torch.cat((z.topk(72,-1).indices, current.long()),-1).sort(-1).values[None]
            self.route_differences+=int((lut.long()!=ref).any().item())
            assert torch.equal(lut.long(),ref),'topk tie/selection changed'
        return lut
    def commit(self,l,k,v,cache):
        if self.variant!='commit_fast':return super().commit(l,k,v,cache)
        x=features(k[0].transpose(0,1).float(),v[0].transpose(0,1),self.bank[l],'response')
        order=balanced(x,[3,3,2,2,2]);xp=reorder(x,order)
        inner=balanced(xp.reshape(-1,64,32),[2,2]).reshape(12,72,64)
        order=order.reshape(12,72,64).gather(2,inner)[:,self.page_order].flatten(1).contiguous()
        from .improve_kernels import commit
        commit(k,v,order,self.weights.get(l),cache,self.pk[l],self.write[l],self.cap)
        self.write[l]=(self.write[l]+72)%self.cap;self.n[l]=min(self.cap,self.n[l]+72);self.commits[l]+=1
    def finish(self):return super().finish()|dict(variant=self.variant,route_differences=self.route_differences)

class AuditKernel:
    def __init__(self,owner,kernel):self.owner=owner;self.kernel=kernel;self.rows=[]
    def __getitem__(self,grid):
        call=self.kernel[grid]
        def run(*args,**kwargs):
            result=call(*args,**kwargs);c=self.owner.engine.current
            if c['chunk'] not in [5,10,20,30,40] or c['step'] not in [0,3] or c['layer'] not in [1,4,9,14,19,24,29]:return result
            q,k,v,scale,topk,lut,lse,out=args[:8]
            assert torch.isfinite(out).all(),'nonfinite sparse output'
            errors=[]
            for head in [0,4,8,11]:
                for qi in [0,63,64,1023,1535,2048,3072,4607]:
                    ids=(lut[0,head,qi//64,:,None].long()*64+torch.arange(64,device=q.device)).flatten()
                    kk=k[0,:,head].index_select(0,ids).float();vv=v[0,:,head].index_select(0,ids).float()
                    target=(q[0,qi,head].float()@kk.T*scale).softmax(-1)@vv
                    err=((out[0,qi,head].float()-target).square().sum()/target.square().sum().clamp_min(1e-8)).item()
                    errors.append(err)
                    assert err<1e-4,(c,head,qi,err)
            self.rows.append(dict(chunk=c['chunk'],step=c['step'],layer=c['layer'],max_nmse=max(errors),mean_nmse=sum(errors)/len(errors)))
            return result
        return run
