"""Inference-only zero-copy KV and removal of training-only route branches."""
import math,torch
from .all_sparse_runtime import AllSparseRuntime
from .all_sparse_attention import Attention,native
from .high_sparse_learn import probe_mass,features,swap_features,correct
from .high_sparse_gain import score
from .high_sparse_learn_kernels import pool
class InferenceRuntime(AllSparseRuntime):
    def decision(self,l,q,pk,base,vvar):
        if self.training_event:return super().decision(l,q,pk,base,vvar)
        m=self.model[l];p=pk.shape[1];keep=int(math.floor(p*.1+1e-6));query=q[0].transpose(0,1).float().reshape(12,72,64,128).index_select(2,self.probe_offsets)
        _,prob,mass=probe_mass(query,pk);feat=features(prob,pk,vvar);own=(torch.arange(72,device=q.device)+p-72)[None].expand(12,-1)
        z=mass.clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices;mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
        sf=swap_features(feat,mass,base.float(),mask);s=score(mass,sf,m.route);chosen,_=correct(mask,s,own)
        lut=chosen.int().topk(keep,-1).indices.sort(-1).values[None].int()
        return lut,None,None,None
    def forward(self,l,module,x,grid_sizes,freqs,cache,start):
        if self.training_event:return super().forward(l,module,x,grid_sizes,freqs,cache,start)
        p=self.n[l];step=self.engine.current['step']
        if not self.sparse or step==4 or p<=72:
            from .high_sparse_gain_runtime import GainRuntime
            return GainRuntime.forward(self,l,module,x,grid_sizes,freqs,cache,start)
        from wan.modules.causal_model import causal_rope_apply
        m=self.model[l];b,s,_=x.shape;h,d=12,128
        q=module.norm_q(module.q(x)).view(b,s,h,d);k=module.norm_k(module.k(x)).view(b,s,h,d);v=module.v(x).view(b,s,h,d)
        q=causal_rope_apply(q,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous();k=causal_rope_apply(k,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        cache['k'][:,p*64:p*64+s].copy_(k);cache['v'][:,p*64:p*64+s].copy_(v)
        kh=cache['k'][:,:p*64+s];vh=cache['v'][:,:p*64+s]
        if self.training_event:
            pk,base,detail=m.pool(kh[0].transpose(0,1).reshape(h,p+72,64,d),vh[0].transpose(0,1).reshape(h,p+72,64,d))
        else:
            ck,cb,cd=m.pool(k[0].transpose(0,1).reshape(h,72,64,d),v[0].transpose(0,1).reshape(h,72,64,d));pk=torch.cat((self.pk[l][:,:p],ck),1);base=torch.cat((self.codes[l][:,:p],cb),1);detail=torch.cat((self.details[l][:,:p],cd),1)
        with torch.no_grad():
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
