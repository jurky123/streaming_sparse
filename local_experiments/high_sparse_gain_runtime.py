"""90% budget deployment: learned route/value and one page-local score correction."""
import math
import torch
from .high_sparse_runtime import HighSparseRuntime
from .improve_runtime import FastRuntime
from .high_sparse_learn import probe_mass,features,route_score,swap_features
from .high_sparse_learn_kernels import pool
class GainRuntime(HighSparseRuntime):
    def __init__(self,engine,checkpoint,strength=1.):
        mode="baseline";self.strength=strength
        assert mode in ['baseline','route','value','joint','swap']
        super().__init__(engine,'global',.9,True);self.learn_mode=mode;self.use_value=mode in ['value','joint','swap']
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu')['models']
        self.uk={};self.uv={};self.rw={};self.sw={};self.vvar={}
        for l,m in saved.items():
            self.uk[l]=m['uk'].cuda().float().contiguous();self.uv[l]=m['uv'].cuda().float().contiguous()
            self.rw[l]=m['route'].cuda().float().contiguous();self.sw[l]=m['swap'].cuda().float().contiguous()
            self.vvar[l]=torch.empty(12,self.cap,device='cuda',dtype=torch.bfloat16)
            if self.use_value:
                self.basis[l]=m['basis'].cuda().to(torch.bfloat16).contiguous();self.centers[l]=m['center'].cuda().float().contiguous()
        self.persistent_bytes+=sum(x.numel()*x.element_size() for group in [self.uk,self.uv,self.rw,self.sw,self.vvar] for x in group.values())
        assert self.persistent_bytes<=self.budget,(self.persistent_bytes,self.budget)
        self.method='gain90_'+str(strength)
    def commit(self,l,k,v,cache):
        start=self.write[l]
        if self.use_value:FastRuntime.commit(self,l,k,v,cache)
        else:super().commit(l,k,v,cache)
        pool(cache,self.basis[l],self.centers[l],self.uk[l],self.uv[l],self.codes[l],self.vvar[l],start=start,ring=True,weighted=self.use_value)
    def route(self,l,q,p):
        device=q.device;cache=self.engine.cache[l]
        var=torch.empty(12,72,device=device,dtype=torch.bfloat16)
        if self.use_value:
            ck=torch.empty(12,72,4,128,device=device);cc=torch.empty(12,72,8,device=device,dtype=torch.bfloat16)
            pool(cache,self.basis[l],self.centers[l],self.uk[l],self.uv[l],cc,var,offset=p*64,weighted=True,pk=ck)
        else:
            ck=cache['k'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,4,16,128).mean(3)
            vm=cache['v'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,64,128).mean(2)
            cc=((vm-self.centers[l][:,None])@self.basis[l].float()).to(torch.bfloat16)
            pool(cache,self.basis[l],self.centers[l],self.uk[l],self.uv[l],cc,var,offset=p*64,weighted=False)
        self.active_pk=torch.cat((self.pk[l][:,:p],ck),1).contiguous();self.active_codes=torch.cat((self.codes[l][:,:p],cc),1).contiguous()
        vv=torch.cat((self.vvar[l][:,:p],var),1).float()
        query=q[0].transpose(0,1).float().reshape(12,72,64,128).index_select(2,torch.tensor([0,21,42,63],device=device))
        _,prob,mass=probe_mass(query,self.active_pk);feat=features(prob,self.active_pk,vv)
        from .high_sparse_gain import score
        from .high_sparse_learn import pick,correct
        own=(torch.arange(72,device=device)+p)[None].expand(12,-1)
        keep=max(1,int(math.floor((p+72)*.1+1e-6)))
        mask=pick(mass,keep,own)
        if self.strength:
            sf=swap_features(feat,mass,self.active_codes.float(),mask)
            mask,changed=correct(mask,score(mass,sf,self.sw[l],self.strength),own)
        ids=mask.int().topk(keep,-1,sorted=False).indices
        lut=ids.sort(-1).values[None].int()
        self.budget_trace[p]=dict(history_pages=p,total_pages=p+72,selected_pages=keep,exact_sparsity=1-keep/(p+72))
        return lut
    def finish(self):return super().finish()|dict(scope='90% sparse calls; original V frozen; bounded logmass correction and at most one swap',gain_strength=self.strength)
