"""All sparse modules train; backbone frozen. Hard grouping has no fictitious sort gradient."""
import math,torch
from .full_route_runtime import FullRouteRuntime
from .all_sparse_model import AllSparse
from .all_sparse_attention import Attention,native
from .high_sparse_learn import probe_mass,features,swap_features,correct
from .high_sparse_gain import score
from .response_hard import balanced
from .cache_grouping import reorder
from .high_sparse_learn_kernels import pool
class AllSparseRuntime(FullRouteRuntime):
    def __init__(self,engine,checkpoint,model=None,probes=8):
        super().__init__(engine,checkpoint);self.probe_count=probes
        old=sum(x.numel()*x.element_size() for bank in [self.bank,self.weights,self.basis,self.centers,self.sw] for x in bank.values())
        self.model=AllSparse(self) if model is None else model
        for l in range(1,30):
            m=self.model[l];self.bank[l]=m.bank;self.weights[l]=m.pool_k;self.basis[l]=m.basis;self.centers[l]=m.center;self.sw[l]=m.route
        self.route_weights={l:self.model[l].route for l in range(1,30)}
        self.details={l:torch.empty(12,self.cap,3,8,device='cuda',dtype=torch.bfloat16) for l in range(1,30)}
        self.original_limit=self.budget;self.budget+=16*2**20
        self.persistent_bytes+=sum(p.numel()*p.element_size() for p in self.model.parameters())-old+sum(x.numel()*x.element_size() for x in self.details.values())
        assert self.persistent_bytes<=self.budget,(self.persistent_bytes,self.budget)
        self.probe_offsets=torch.linspace(0,63,probes,device='cuda').round().long();self.persistent_bytes+=probes*8
        assert self.persistent_bytes<=self.budget
    def commit(self,l,k,v,cache):
        m=self.model[l];kf=k[0].transpose(0,1).float();vf=v[0].transpose(0,1)
        x=m.group_features(kf[:,None],vf.float()[:,None]).squeeze(1);order=balanced(x,[3,3,2,2,2]);xp=reorder(x,order)
        inner=balanced(xp.reshape(-1,64,32),[2,2]).reshape(12,72,64);order=order.reshape(12,72,64).gather(2,inner)[:,self.page_order].flatten(1)
        kr=reorder(kf,order);vr=reorder(vf,order);pk,base,detail=m.pool(kr.reshape(12,72,64,128),vr.reshape(12,72,64,128))
        slots=(torch.arange(72,device=k.device)+self.write[l])%self.cap;ids=(slots[:,None]*64+self.offsets).flatten()
        cache['k'][0].index_copy_(0,ids,kr.transpose(0,1).to(torch.bfloat16));cache['v'][0].index_copy_(0,ids,vr.transpose(0,1))
        self.pk[l].index_copy_(1,slots,pk);self.codes[l].index_copy_(1,slots,base);self.details[l].index_copy_(1,slots,detail)
        pool(cache,self.basis[l],self.centers[l],self.uk[l],self.uv[l],self.codes[l],self.vvar[l],start=self.write[l],ring=True,weighted=False)
        self.write[l]=(self.write[l]+72)%self.cap;self.n[l]=min(self.cap,self.n[l]+72);self.commits[l]+=1
    def decision(self,l,q,pk,base,vvar):
        m=self.model[l];p=pk.shape[1];keep=int(math.floor(p*.1+1e-6));query=q[0].transpose(0,1).float().reshape(12,72,64,128).index_select(2,self.probe_offsets)
        _,prob,mass=probe_mass(query,pk);feat=features(prob,pk,vvar);own=(torch.arange(72,device=q.device)+p-72)[None].expand(12,-1)
        z=mass.detach().clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices;mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
        sf=swap_features(feat,mass,base.float(),mask);s=score(mass,sf,m.route)
        with torch.no_grad():
            rem=s.masked_fill(~mask,float('inf'));rem.scatter_(-1,own[...,None],float('inf'));low,drop=rem.min(-1);high,add=s.masked_fill(mask,-float('inf')).max(-1)
            new,_=correct(mask,s,own);alt=mask.clone().scatter_(-1,drop[...,None],False).scatter_(-1,add[...,None],True)
            def lut(x):return x.int().topk(keep,-1).indices.sort(-1).values[None].int()
            chosen=lut(new);baseids=lut(mask);altids=lut(alt)
        margin=s.gather(-1,add[...,None]).squeeze(-1)-s.gather(-1,drop[...,None]).squeeze(-1)
        return chosen,baseids,altids,margin
    def forward(self,l,module,x,grid_sizes,freqs,cache,start):
        p=self.n[l];step=self.engine.current['step']
        if not self.sparse or step==4 or p<=72:
            from .high_sparse_gain_runtime import GainRuntime
            return GainRuntime.forward(self,l,module,x,grid_sizes,freqs,cache,start)
        from wan.modules.causal_model import causal_rope_apply
        m=self.model[l];b,s,_=x.shape;h,d=12,128
        q=module.norm_q(module.q(x)).view(b,s,h,d);k=module.norm_k(module.k(x)).view(b,s,h,d);v=module.v(x).view(b,s,h,d)
        q=causal_rope_apply(q,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous();k=causal_rope_apply(k,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        kh=torch.cat((cache['k'][:,:p*64].detach(),k),1);vh=torch.cat((cache['v'][:,:p*64].detach(),v),1)
        if self.training_event:
            pk,base,detail=m.pool(kh[0].transpose(0,1).reshape(h,p+72,64,d),vh[0].transpose(0,1).reshape(h,p+72,64,d))
        else:
            ck,cb,cd=m.pool(k[0].transpose(0,1).reshape(h,72,64,d),v[0].transpose(0,1).reshape(h,72,64,d));pk=torch.cat((self.pk[l][:,:p],ck),1);base=torch.cat((self.codes[l][:,:p],cb),1);detail=torch.cat((self.details[l][:,:p],cd),1)
        with torch.no_grad():
            cache['k'][:,p*64:p*64+s].copy_(k);cache['v'][:,p*64:p*64+s].copy_(v)
            var=torch.empty(h,72,device=x.device,dtype=torch.bfloat16);pool(cache,m.basis,m.center,self.uk[l],self.uv[l],base,var,offset=p*64,weighted=False)
            vv=torch.cat((self.vvar[l][:,:p],var),1).float()
        lut,baseids,altids,margin=self.decision(l,q,pk,base,vv);basis=m.basis.to(torch.bfloat16)
        if self.training_event:
            out=Attention.apply(q,kh,vh,pk,base,detail,basis,m.center,m.comp_logscale,lut,self.bare_kernel)
            with torch.no_grad():
                old=native(q,kh,vh,pk,base,detail,basis,m.center,m.comp_logscale,baseids,self.bare_kernel);alt=native(q,kh,vh,pk,base,detail,basis,m.center,m.comp_logscale,altids,self.bare_kernel);delta=alt.float()-old.float()
            gate=margin.sigmoid()[:,:,None,None].expand(h,72,64,d).reshape(h,s,d).transpose(0,1)[None];proxy=(gate*delta).to(out.dtype);out=out+(proxy-proxy.detach())
        else:out=native(q,kh,vh,pk,base,detail,basis,m.center,m.comp_logscale,lut,self.bare_kernel)
        with torch.no_grad():cache['global_end_index'].fill_(start+s);cache['local_end_index'].fill_(p*64+s)
        self.budget_trace[p]=dict(history_pages=p,total_pages=p+72,selected_pages=lut.shape[-1],exact_sparsity=1-lut.shape[-1]/(p+72))
        return module.o(out.flatten(2))
    def finish(self):return super().finish()|dict(scope="all sparse modules trained; frozen backbone; fixed hard page budget", current_summary="page-local learned paired K/V pooling, four representatives", route_implementation="native PyTorch probes/features and one set-conditioned swap; not yet fused", gradient="all sparse parameters; conditional attention backward plus pair-margin STE", all_sparse_parameters=sum(p.numel() for p in self.model.parameters()),parameter_families=list(dict(self.model[1].named_parameters())),original_limit_bytes=self.original_limit,extra_limit_bytes=16*2**20,gradient_scope='all sparse families get output-loss gradients through page-local pooling/route/compensation; physical sort boundaries remain hard and detached; no backbone training')
