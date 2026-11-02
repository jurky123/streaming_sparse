"""Page-local rank8 V payload, immutable clean commits, exact original sparse route."""
import torch
from .improve_runtime import FastRuntime
from .compensation_kernels import compensate,reference

class WrappedKernel:
    def __init__(self,rt,base):self.rt=rt;self.base=base
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            self.base[grid](*args,**kwargs)
            r=self.rt;l=r.active_layer
            if not r.enabled[l]:return
            q=args[0];lut=args[5];lse=args[6];out=args[7];p=r.n[l]
            result,_,_=compensate(q,r.pk[l],r.codes[l],lut,out,lse,r.basis[l],r.centers[l],r.gamma[l],p)
            if r.comp_audit and r.engine.current['chunk'] in [2,5,20,40] and r.engine.current['step'] in [0,3]:
                # Real Q/K-summary/V-code checks at early and late chunks, all enabled layers/heads.
                tile=35;sl=slice(tile*64,(tile+1)*64)
                target,_,_=reference(q[:,sl].contiguous(),r.pk[l],r.codes[l],lut[:,:,tile:tile+1].contiguous(),out[:,sl].contiguous(),lse[:,:,sl].contiguous(),r.basis[l],r.centers[l],r.gamma[l],p)
                error=(result[:,sl].float()-target.float()).square().sum()/target.float().square().sum().clamp_min(1e-8)
                assert error<1e-5,(l,r.engine.current,error.item())
                assert torch.isfinite(result).all()
                r.comp_checks.append(dict(layer=l,chunk=r.engine.current['chunk'],step=r.engine.current['step'],nmse=error.item()))
            out.copy_(result)
        return call

class CompensationRuntime(FastRuntime):
    def __init__(self,engine,checkpoint,selection=None,mode='guarded',audit=False):
        super().__init__(engine,'route_fast');self.method='compensation_'+mode;self.comp_checks=[];self.active_layer=None;self.comp_audit=audit
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu')['models'];self.codes={};self.basis={};self.centers={};self.gamma={};self.enabled={}
        for l,m in saved.items():
            self.basis[l]=m['basis'].cuda().to(torch.bfloat16).contiguous()
            self.centers[l]=m['center'].cuda().float().contiguous()
            values=[1.]*12 if mode=='raw' else [0.]*12 if mode=='zero' else selection['gamma'][str(l)]
            self.gamma[l]=torch.tensor(values,device='cuda',dtype=torch.float32)
            self.enabled[l]=any(x>0 for x in values)
            self.codes[l]=torch.empty(12,self.cap,8,device='cuda',dtype=torch.bfloat16)
        self.persistent_bytes+=sum(x.numel()*x.element_size() for group in [self.basis,self.centers,self.gamma,self.codes] for x in group.values())
        assert self.persistent_bytes<=self.budget,(self.persistent_bytes,self.budget)
        self.kernel=WrappedKernel(self,self.kernel)
    def forward(self,l,*args,**kwargs):
        self.active_layer=l
        return super().forward(l,*args,**kwargs)
    def commit(self,l,k,v,cache):
        start=self.write[l]
        super().commit(l,k,v,cache)
        if not self.enabled[l]:return
        slots=(torch.arange(72,device=k.device)+start)%self.cap
        ids=(slots[:,None]*64+self.offsets).flatten()
        values=cache['v'][0].index_select(0,ids).transpose(0,1).float().reshape(12,72,64,128).mean(2)
        code=(values-self.centers[l][:,None])@self.basis[l].float()
        self.codes[l].index_copy_(1,slots,code.to(torch.bfloat16))
    def reset(self):
        super().reset();self.comp_checks=[]
    def finish(self):
        return super().finish()|dict(compensation_checks=self.comp_checks,enabled_heads=sum(int((g>0).sum().item()) for g in self.gamma.values()),variant=self.method)
