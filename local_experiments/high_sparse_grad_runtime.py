"""Genuine x0 backprop to last-layer router with a hard-forward swap STE."""
import torch
from .high_sparse_gain_runtime import GainRuntime
from .high_sparse_gain import score
from .high_sparse_learn import probe_mass,features,swap_features,pick
from .high_sparse_runtime import compensate

class GradKernel:
    def __init__(self,rt,base):self.rt=rt;self.base=base
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            r=self.rt;active=r.training_event and r.active_layer==29
            if not active:return self.base[grid](*args,**kwargs)
            pk,code=r.active_pk,r.active_codes
            self.base[grid](*args,**kwargs)
            mass,feat,mask,own=r.meta
            with torch.no_grad():
                s=score(mass,feat,r.weight)
                rem=s.masked_fill(~mask,float('inf'));rem.scatter_(-1,own[...,None],float('inf'))
                drop=rem.argmin(-1);add=s.masked_fill(mask,-float('inf')).argmax(-1)
                alt=mask.clone().scatter_(-1,drop[...,None],False).scatter_(-1,add[...,None],True)
                outs=[]
                for selected in [mask,alt]:
                    ids=selected.int().topk(args[5].shape[-1],-1,sorted=False).indices.sort(-1).values[None].int()
                    a=list(args);a[4]=ids.shape[-1];a[5]=ids;a[6]=torch.empty_like(args[6]);a[7]=torch.empty_like(args[7])
                    self.base.base[grid](*a,**kwargs)
                    compensate(a[0],pk,code,ids,a[7],a[6],r.basis[29],r.centers[29],r.gamma[29]);outs.append(a[7])
                gate=s.gather(-1,add[...,None]).squeeze(-1)>s.gather(-1,drop[...,None]).squeeze(-1)
                expanded=gate[:,:,None,None].expand(12,72,64,128).reshape(12,4608,128).transpose(0,1)[None]
                expected=torch.where(expanded,outs[1],outs[0])
                assert torch.equal(args[7],expected),'Hard native forward differs from per-tile two-branch selection'
                r.proxy_data=(mass,feat,add,drop,outs[1].float()-outs[0].float())
                r.native_checks+=1
        return call

class GradRuntime(GainRuntime):
    def __init__(self,engine,checkpoint,enabled=True):
        super().__init__(engine,checkpoint,0.)
        self.enabled_last=enabled;self.training_event=False;self.meta=None;self.proxy_data=None;self.last_margin=None;self.native_checks=0
        self.weight=torch.nn.Parameter(self.sw[29]);self.sw[29]=self.weight
        self.kernel=GradKernel(self,self.kernel)
        self.handle=engine.wrapper.model.blocks[29].self_attn.register_forward_hook(self.inject)
    def route(self,l,q,p):
        self.strength=1. if self.enabled_last and l==29 else 0.
        lut=super().route(l,q,p)
        if self.training_event and l==29:
            # Recompute only the features needed by the training proxy. Historical
            # variance is immutable; current variance uses the same fused kernel.
            from .high_sparse_learn_kernels import pool
            var=torch.empty(12,72,device=q.device,dtype=torch.bfloat16)
            pool(self.engine.cache[l],self.basis[l],self.centers[l],self.uk[l],self.uv[l],self.active_codes,var,offset=p*64,weighted=False)
            vv=torch.cat((self.vvar[l][:,:p],var),1).float()
            query=q[0].transpose(0,1).float().reshape(12,72,64,128).index_select(2,torch.tensor([0,21,42,63],device=q.device))
            _,prob,mass=probe_mass(query,self.active_pk);f=features(prob,self.active_pk,vv)
            own=(torch.arange(72,device=q.device)+p)[None].expand(12,-1);mask=pick(mass,lut.shape[-1],own)
            self.meta=(mass,swap_features(f,mass,self.active_codes.float(),mask),mask,own)
        return lut
    def inject(self,module,args,output):
        if not self.training_event or self.proxy_data is None:return output
        mass,feat,add,drop,delta=self.proxy_data
        with torch.enable_grad():
            s=score(mass,feat,self.weight)
            margin=(s.gather(-1,add[...,None])-s.gather(-1,drop[...,None])).squeeze(-1)
            self.last_margin=margin;margin.retain_grad()
            gate=margin.sigmoid()
            gate=gate[:,:,None,None].expand(12,72,64,128).reshape(12,4608,128).transpose(0,1)[None]
            proxy=module.o((gate*delta).flatten(2).to(output.dtype))
            injected=output.detach()+(proxy-proxy.detach())
            assert torch.equal(injected.detach(),output.detach()),'STE changed native self-attention forward'
            return injected
    def clear_proxy(self):self.meta=None;self.proxy_data=None;self.last_margin=None
    def close(self):
        self.handle.remove();self.clear_proxy();super().close()
    def finish(self):return super().finish()|dict(trainable_route_layer=29,trainable_route_parameters=168,gradient='hard native forward; sigmoid pair margin STE backward; frozen prefix, candidates and topk indices',native_branch_checks=self.native_checks,last_layer_enabled=self.enabled_last)
