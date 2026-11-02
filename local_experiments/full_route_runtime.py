"""All 29 sparse layers: native forward, complete conditional attention backward."""
import torch,types
from torch.utils.checkpoint import checkpoint
from .route_graph_runtime import GraphRuntime
from .full_route_attention import NativeAttention,native

class FullRouteRuntime(GraphRuntime):
    def __init__(self,engine,model,checkpoint_blocks=True):
        super().__init__(engine,model,1.)
        self.training_event=False;self.checkpoint_blocks=checkpoint_blocks
        self.route_weights={l:torch.nn.Parameter(self.sw[l]) for l in range(1,30)}
        self.sw.update(self.route_weights);self.original_blocks=[b.forward for b in engine.wrapper.model.blocks]
        self.bare_kernel=self.kernel.base;self.branch_checks=0
        for i,b in enumerate(engine.wrapper.model.blocks):
            original=self.original_blocks[i]
            def wrapped(module,*args,index=i,original=original,**kwargs):
                if self.training_event and self.checkpoint_blocks and index>0 and torch.is_grad_enabled():
                    return checkpoint(original,*args,**kwargs,use_reentrant=False)
                return original(*args,**kwargs)
            b.forward=types.MethodType(wrapped,b)
    def forward(self,l,module,x,grid_sizes,freqs,cache,start):
        if not self.training_event:return super().forward(l,module,x,grid_sizes,freqs,cache,start)
        from wan.modules.causal_model import causal_rope_apply
        self.active_layer=l;p=self.n[l];step=self.engine.current['step'];assert step<4 and p>72
        b,s,_=x.shape;h,d=12,128
        q=module.norm_q(module.q(x)).view(b,s,h,d);k=module.norm_k(module.k(x)).view(b,s,h,d);v=module.v(x).view(b,s,h,d)
        q=causal_rope_apply(q,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        k=causal_rope_apply(k,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        with torch.no_grad():
            cache['k'][:,p*64:p*64+s].copy_(k);cache['v'][:,p*64:p*64+s].copy_(v)
            lut=self.route(l,q,p).clone()
            mass,sf,mask,own,add,drop=self.route_meta
            # Only the two shortlisted feature vectors need to survive graph reuse.
            fa=sf.gather(2,add[...,None,None].expand(-1,-1,1,14)).squeeze(2).clone()
            fd=sf.gather(2,drop[...,None,None].expand(-1,-1,1,14)).squeeze(2).clone()
            la=mass.gather(-1,add[...,None]).squeeze(-1).clamp_min(1e-12).log().clone()
            ld=mass.gather(-1,drop[...,None]).squeeze(-1).clamp_min(1e-12).log().clone()
            baseids=mask.int().topk(lut.shape[-1],-1).indices.sort(-1).values[None].int()
            altmask=mask.clone().scatter_(-1,drop[...,None],False).scatter_(-1,add[...,None],True)
            altids=altmask.int().topk(lut.shape[-1],-1).indices.sort(-1).values[None].int()
            pk_native=self.active_pk.detach();code_native=self.active_codes.detach().clone()
        kh=torch.cat((cache['k'][:,:p*64].detach(),k),1);vh=torch.cat((cache['v'][:,:p*64].detach(),v),1)
        ck=k[0].transpose(0,1).float().reshape(h,72,4,16,d).mean(3)
        vm=v[0].transpose(0,1).float().reshape(h,72,64,d).mean(2)
        cc=((vm-self.centers[l][:,None])@self.basis[l].float()).to(torch.bfloat16)
        # Preserve exact deployed current summaries; attach their linear pooling VJP.
        pk=torch.cat((pk_native[:,:p],pk_native[:,p:]+(ck-ck.detach())),1)
        code=torch.cat((code_native[:,:p],code_native[:,p:]+(cc-cc.detach())),1)
        out=NativeAttention.apply(q,kh,vh,pk,code,self.basis[l],self.centers[l],lut,self.bare_kernel,self.gamma[l])
        with torch.no_grad():
            base=native(q,kh,vh,pk,code,self.basis[l],self.centers[l],baseids,self.bare_kernel,self.gamma[l])
            alt=native(q,kh,vh,pk,code,self.basis[l],self.centers[l],altids,self.bare_kernel,self.gamma[l])
            delta=alt.float()-base.float()
            # A head/tile's output must equal exactly one of these two branches.
            chosen=(lut[0]==add[...,None]).any(-1)
            expanded=chosen[:,:,None,None].expand(h,72,64,d).reshape(h,s,d).transpose(0,1)[None]
            assert torch.equal(out.detach(),torch.where(expanded,alt,base))
            self.branch_checks+=1
        w=self.route_weights[l]
        margin=la+torch.tanh(torch.einsum('htf,hf->ht',fa,w))-ld-torch.tanh(torch.einsum('htf,hf->ht',fd,w))
        gate=margin.sigmoid()[:,:,None,None].expand(h,72,64,d).reshape(h,s,d).transpose(0,1)[None]
        proxy=(gate*delta).to(out.dtype)
        # Do NOT detach native output: this carries Q/K/V and summary gradients
        # through all later layers, unlike the last-layer-only pilot.
        out=out+(proxy-proxy.detach())
        with torch.no_grad():
            cache['global_end_index'].fill_(start+s);cache['local_end_index'].fill_(p*64+s)
        self.active_pk=None;self.active_codes=None;self.route_meta=None
        return module.o(out.flatten(2))
    def close(self):
        for b,original in zip(self.engine.wrapper.model.blocks,self.original_blocks):b.forward=original
        super().close()
    def finish(self):return super().finish()|dict(trainable_layers=list(range(1,30)),trainable_route_parameters=29*12*14,checkpoint_blocks=self.checkpoint_blocks,native_branch_checks=self.branch_checks,gradient='native attention forward; FP32 efficient-SDPA recompute backward for Q/K/V/current summaries; fixed hard mask + pair-margin STE for route weights; frozen historical KV')
