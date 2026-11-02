"""Graph current-summary computation directly from existing current KV cache."""
import time,types,torch
from .gap_cached_ops import cached_pool
from .gap_runtime import GapRuntime
from .gap_graph_runtime import GapGraphRuntime
from .all_sparse_attention import Attention,native
from .high_sparse_learn_kernels import pool
class GapFastRuntime(GapGraphRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # These old weighted-value parameters are unused by every gap-model path.
        # Native variance pooling has WEIGHTED=False and never reads UK/UV.
        released=sum(x.numel()*x.element_size() for group in [self.uk,self.uv,self.rw] for x in group.values())
        self.uk=self.centers;self.uv=self.centers;self.rw={};self.persistent_bytes-=released
        self.current_outputs=(torch.empty(12,72,4,128,device='cuda'),torch.empty(12,72,8,device='cuda',dtype=torch.bfloat16),torch.empty(12,72,3,8,device='cuda',dtype=torch.bfloat16),torch.empty(12,72,8,device='cuda',dtype=torch.bfloat16))
        self.current_aux=sum(x.numel()*x.element_size() for x in self.current_outputs);self.persistent_bytes+=self.current_aux
        assert self.persistent_bytes<=self.budget,(self.persistent_bytes,self.budget)
        for m in self.model.layers.values():
            m._cached_haar=self.ghaar;m.pool=types.MethodType(cached_pool,m)
        self.pool_graphs={};self.pool_capture_seconds=0.;self.summary_pool=torch.cuda.graph_pool_handle()
    @torch.no_grad()
    def current_summary(self,l):
        p=self.n[l];key=(l,p);cache=self.engine.cache[l];m=self.model[l]
        if key not in self.pool_graphs:
            k=cache['k'][0,p*64:p*64+4608].transpose(0,1).reshape(12,72,64,128)
            v=cache['v'][0,p*64:p*64+4608].transpose(0,1).reshape_as(k)
            def run():
                pk,base,detail=m.pool(k,v,cheap=True);cov=m.covariance(k,v)
                for dest,src in zip(self.current_outputs,(pk,base,detail,cov)):dest.copy_(src)
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):run()
            torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.summary_pool):run()
            self.pool_graphs[key]=graph;self.pool_capture_seconds+=time.perf_counter()-start
        self.pool_graphs[key].replay();return self.current_outputs
    def forward(self,l,module,x,grid_sizes,freqs,cache,start):
        if self.training_event:return GapRuntime.forward(self,l,module,x,grid_sizes,freqs,cache,start)
        p=self.n[l];step=self.engine.current['step']
        if not self.sparse or step==4 or p<=72:
            from .high_sparse_gain_runtime import GainRuntime
            return GainRuntime.forward(self,l,module,x,grid_sizes,freqs,cache,start)
        from wan.modules.causal_model import causal_rope_apply
        m=self.model[l];b,s,_=x.shape;h,d=12,128
        q=module.norm_q(module.q(x)).view(b,s,h,d);k=module.norm_k(module.k(x)).view(b,s,h,d);v=module.v(x).view(b,s,h,d)
        q=causal_rope_apply(q,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous();k=causal_rope_apply(k,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        if self.training_event:
            kh=torch.cat((cache['k'][:,:p*64].detach(),k),1);vh=torch.cat((cache['v'][:,:p*64].detach(),v),1)
        else:
            cache['k'][:,p*64:p*64+s].copy_(k);cache['v'][:,p*64:p*64+s].copy_(v)
            kh=cache['k'][:,:p*64+s];vh=cache['v'][:,:p*64+s]
        current_k=k[0].transpose(0,1).reshape(h,72,64,d);current_v=v[0].transpose(0,1).reshape(h,72,64,d)
        ck,cb,cd,ccov=self.current_summary(l)
        if self.training_event:
            hk=kh[0,:p*64].transpose(0,1).reshape(h,p,64,d);hv=vh[0,:p*64].transpose(0,1).reshape(h,p,64,d)
            hp,hb,hd=m.pool(hk,hv);hc=m.covariance(hk,hv)
        else:hp,hb,hd,hc=self.pk[l][:,:p],self.codes[l][:,:p],self.details[l][:,:p],self.cov[l][:,:p]
        pk=torch.cat((hp,ck),1);base=torch.cat((hb,cb),1);detail=torch.cat((hd,cd),1);cov=torch.cat((hc,ccov),1)
        with torch.no_grad():
            cache['k'][:,p*64:p*64+s].copy_(k);cache['v'][:,p*64:p*64+s].copy_(v)
            var=torch.empty(h,72,device=x.device,dtype=torch.bfloat16);pool(cache,m.basis,m.center,self.uk[l],self.uv[l],base,var,offset=p*64,weighted=False)
            vv=torch.cat((self.vvar[l][:,:p],var),1).float()
        lut,baseids,altids,margin=self.decision(l,q,pk,base,detail,cov,vv);basis=m.basis.to(torch.bfloat16)
        if self.training_event:
            out=Attention.apply(q,kh,vh,pk,base,detail,basis,m.center,m.comp_logscale,lut,self.bare_kernel)
            with torch.no_grad():
                old=native(q,kh,vh,pk,base,detail,basis,m.center,m.comp_logscale,baseids,self.bare_kernel);alt=native(q,kh,vh,pk,base,detail,basis,m.center,m.comp_logscale,altids,self.bare_kernel);delta=alt.float()-old.float()
            gate=margin.sigmoid()[:,:,None,None].expand(h,72,64,d).reshape(h,s,d).transpose(0,1)[None];proxy=(gate*delta).to(out.dtype);out=out+(proxy-proxy.detach())
        else:out=native(q,kh,vh,pk,base,detail,basis,m.center,m.comp_logscale,lut,self.bare_kernel)
        with torch.no_grad():cache['global_end_index'].fill_(start+s);cache['local_end_index'].fill_(p*64+s)
        self.budget_trace[p]=dict(history_pages=p,total_pages=p+72,selected_pages=lut.shape[-1],exact_sparsity=1-lut.shape[-1]/(p+72))
        return module.o(out.flatten(2))
    def finish(self):return super().finish()|dict(current_pool_implementation='CUDA graphs read existing KV cache, shared external outputs, no raw KV duplicate',current_output_bytes=self.current_aux,current_graphs=len(self.pool_graphs),current_capture_seconds=self.pool_capture_seconds)
