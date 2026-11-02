import torch
from .sparse_expansion_fast_runtime import ExpansionRuntime
from .conditional_v_kernels import details_from_values,compensate
class ConditionalKernel:
    def __init__(self,rt,base):self.rt=rt;self.base=base
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            r=self.rt;l=r.active_layer;a=list(args);a[4]=a[5].shape[-1];self.base[grid](*a,**kwargs)
            compensate(a[0],r.active_pk,r.active_codes,r.active_detail,r.active_choice,a[5],a[7],a[6],r.basis[l],r.centers[l],r.gamma[l])
            r.active_pk=None;r.active_codes=None;r.active_detail=None;r.active_choice=None
        return call
class ConditionalRuntime(ExpansionRuntime):
    def __init__(self,engine,checkpoint,probes=4,steps=1,mode='paired4'):
        assert mode in ['paired4','haar1'];super().__init__(engine,checkpoint,probes,steps);self.detail_mode=mode
        self.details={l:torch.empty((12,self.cap,3,8) if mode=='paired4' else (12,self.cap,8),device='cuda',dtype=torch.bfloat16) for l in range(1,30)}
        self.choices={l:torch.empty(12,self.cap,device='cuda',dtype=torch.uint8) for l in range(1,30)} if mode=='haar1' else {}
        self.extra_bytes=sum(x.numel()*x.element_size() for bank in [self.details,self.choices] for x in bank.values());self.original_limit=self.budget;self.budget+=16*2**20;self.persistent_bytes+=self.extra_bytes
        assert self.persistent_bytes<=self.budget
        self.kernel=ConditionalKernel(self,self.kernel.base);self.active_detail=None;self.active_choice=None
    def commit(self,l,k,v,cache):
        start=self.write[l];super().commit(l,k,v,cache)
        slots=(torch.arange(72,device=k.device)+start)%self.cap;ids=(slots[:,None]*64+self.offsets).flatten()
        values=cache['v'][0].index_select(0,ids).transpose(0,1).reshape(12,72,64,128)
        detail=details_from_values(values,self.basis[l],self.centers[l])
        if self.detail_mode=='haar1':
            choice=detail.float().square().sum(-1).argmax(-1);detail=detail.gather(2,choice[...,None,None].expand(-1,-1,1,8)).squeeze(2);self.choices[l].index_copy_(1,slots,choice.to(torch.uint8))
        self.details[l].index_copy_(1,slots,detail)
    def route(self,l,q,p):
        lut=super().route(l,q,p);values=self.engine.cache[l]['v'][0,p*64:p*64+4608].transpose(0,1).reshape(12,72,64,128)
        detail=details_from_values(values,self.basis[l],self.centers[l])
        if self.detail_mode=='haar1':
            choice=detail.float().square().sum(-1).argmax(-1);detail=detail.gather(2,choice[...,None,None].expand(-1,-1,1,8)).squeeze(2)
            self.active_choice=torch.cat((self.choices[l][:,:p],choice.to(torch.uint8)),1).contiguous()
        else:self.active_choice=self.router.layer # Unused by paired kernel, no extra array.
        self.active_detail=torch.cat((self.details[l][:,:p],detail),1).contiguous();return lut
    def finish(self):return super().finish()|dict(conditional_v=self.detail_mode,extra_history_bytes=self.extra_bytes,original_limit_bytes=self.original_limit,allowed_relaxation_bytes=16*2**20)
