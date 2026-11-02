"""Fused four-representative normalizers; preserves native reduction orders."""
import math,time,torch,triton as tr
from .gap_runtime import GapRuntime
from .all_sparse_attention import Attention,native
from .high_sparse_learn_kernels import pool
from .gap_feature_kernel import _features_conditional
from .route_fused_kernels import _lut
from .gap_graph_runtime import route_features
from .high_sparse_learn import correct
from .high_sparse_gain import score
from .gap_repstats import stats
from .gap_fast_runtime import GapFastRuntime
from . import gap_graph_runtime as graph_module

def postprocess(logits,qp,base,detail,cov,vvar,kvar,w,gate,haar,training=False):
    h,t,n,p,_=logits.shape;keep=int(math.floor(p*.1+1e-6))
    logz,within=stats(logits,order=1,div=1,exp=0);prob=logz.softmax(-1);mass=prob.mean(2)
    weight=(prob[...,None]*within).mean(2)/mass[...,None].clamp_min(1e-12)
    coefficients=torch.einsum('htpr,ar->htpa',weight,haar)
    conditional=base.float()[:,None]+torch.einsum('htpa,hpad->htpd',coefficients,detail.float())
    # First-order diagonal projected KV-covariance proxy, not an exact page oracle.
    qbar=torch.einsum('htmp,htmd->htpd',prob,qp)/(n*mass[...,None].clamp_min(1e-12))
    residual=qbar*cov.float()[:,None]
    own=(torch.arange(t,device=logits.device)+p-t)[None].expand(h,-1)
    z=mass.detach().clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=(weighted[...,None]*conditional).sum(2)/den;allvalue=(mass[...,None]*conditional).sum(2)
    spread=prob.clamp_min(1e-12).log().std(2,unbiased=False).clamp(0,5)/5
    feat=torch.empty(h,t,p,14,device=logits.device)
    _features_conditional[(h*t,)](prob.max(2).values,mass,conditional,vvar,kvar,w,mask,den,selected,allvalue,(mass/den).clamp(0,2),spread,feat,p,tr.next_power_of_2(p),*conditional.stride()[:3],num_warps=4,enable_fp_fusion=False)
    s=score(mass,feat,w)
    contribution=mass[...,None]*residual;error=(contribution*(~mask)[...,None]).sum(2)
    # Add reduces predicted error by 2<E,e>-||e||²; dropping costs 2<E,e>+||e||².
    residual_energy=error.square().sum(-1,keepdim=True).clamp_min(1e-8)
    gain=(2*(error[:,:,None]*contribution).sum(-1)+(2*mask.float()-1)*contribution.square().sum(-1))/residual_energy
    s=s+gate.tanh()[:,None,None]*gain.clamp(-5,5)
    assert not training
    chosen=torch.empty(1,h,t,keep,device=logits.device,dtype=torch.int32)
    _lut[(h*t,)](s,mask,ids,chosen,p,keep,tr.next_power_of_2(p),tr.next_power_of_2(keep),num_warps=4)
    return chosen,None,None,None

class GapFeatureRuntime(GapFastRuntime):
    @torch.no_grad()
    def decision(self,l,q,pk,base,detail,cov,vvar):
        if self.training_event:return super().decision(l,q,pk,base,detail,cov,vvar)
        m=self.model[l];h,p,_,d=pk.shape;t=q.shape[1]//64;n=self.probe_count
        query=q[0].transpose(0,1).float().reshape(h,t,64,d).index_select(2,self.probe_offsets)
        raw=self.graw[:h*t*n*p*4].view(h,t*n,p*4)
        torch.bmm(query.reshape(h,t*n,d),pk.flatten(1,2).transpose(1,2),out=raw)
        torch.div(raw,math.sqrt(d),out=raw);logits=raw.view(h,t,n,p,4)
        qp=query@m.basis.to(torch.bfloat16).float()[:,None]/math.sqrt(d);kvar=(pk-pk.mean(2,keepdim=True)).square().mean((2,3)).log1p()
        args=(self.graw[:logits.numel()].view_as(logits),self.gqp,self.gbase[:base.numel()].view_as(base),self.gdetail[:detail.numel()].view_as(detail),self.gcov[:cov.numel()].view_as(cov),self.gvar[:vvar.numel()].view_as(vvar),self.gkvar[:kvar.numel()].view_as(kvar),self.gw,self.ggate,self.ghaar)
        for dest,src in zip(args[1:-1],(qp,base,detail,cov,vvar,kvar,m.route,m.residual_gate)):dest.copy_(src)
        if p not in self.graphs_new:
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=postprocess(*args)
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.graph_pool):out=postprocess(*args)
            self.graphs_new[p]=graph;self.outputs_new[p]=out;self.capture_seconds+=time.perf_counter()-start
        self.graphs_new[p].replay();return self.outputs_new[p]
    def finish(self):return super().finish()|dict(normalizers='fused normalization, conditional features, swap LUT')
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
