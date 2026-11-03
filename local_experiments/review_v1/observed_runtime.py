"""Instrumented quality runtime: no global replacements, student-state labels."""
import math,torch
from .runtime import StreamingRuntime,GroupingOps,SelectedAttention
from .ops import SummaryOps,Summary,RouteOps

class ObservedRuntime(StreamingRuntime):
    def __init__(self,*args,observer=None,**kwargs):
        self.observer=observer
        super().__init__(*args,**kwargs)
    def forward(self,l,module,x,grid_sizes,freqs,cache,start):
        from wan.modules.causal_model import causal_rope_apply
        store=self.stores[l];m=self.model[l];p=store.n;b,n,_=x.shape
        assert (b,n)==(1,4608)
        q=module.norm_q(module.q(x)).view(b,n,12,128);k=module.norm_k(module.k(x)).view_as(q);v=module.v(x).view_as(q)
        q=causal_rope_apply(q,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        k=causal_rope_apply(k,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        # History detached, current gradients retained; no implicit crosschunk BPTT.
        kh=torch.cat((cache['k'][:,:p*64].detach(),k),1);vh=torch.cat((cache['v'][:,:p*64].detach(),v),1)
        step=self.engine.current['step']
        if not self.sparse or step==4 or p<=72:out=self.engine.dense(q,kh,vh)
        else:
            current=SummaryOps.build(m,k[0].transpose(0,1).reshape(12,72,64,128),v[0].transpose(0,1).reshape(12,72,64,128),cheap=True)
            history=store.summary.slice(0,p)
            if torch.is_grad_enabled():
                history=SummaryOps.build(m,kh[0,:p*64].transpose(0,1).reshape(12,p,64,128),vh[0,:p*64].transpose(0,1).reshape(12,p,64,128))
            summary=Summary(*(torch.cat((getattr(history,name),getattr(current,name)),1) for name in ('keys','mean','detail','stats')))
            query=q[0].transpose(0,1);offsets=torch.linspace(0,63,self.config.probes,device=q.device).round().long()
            # Process tiles separately: bounded feature lifetime, no full probe-page buffer.
            scores=[]
            with torch.no_grad():
                for tile in range(0,72,8):
                    probes=query[:,tile*64:(tile+8)*64].reshape(12,8,64,128).index_select(2,offsets).flatten(1,2)
                    f,_=RouteOps.features(probes,summary)
                    scores.append((f*m.theta[:,None,None]).sum(-1).reshape(12,8,self.config.probes,p+72).mean(2))
                scores=torch.cat(scores,1);own=(torch.arange(72,device=q.device)+p)[None].expand(12,-1)
                threshold=None if self.thresholds is None else self.thresholds[l,:,step][:,None,None]
                selection=RouteOps.select(scores,own,self.config,threshold)
                if threshold is None:selection['fixed_count']=max(1,math.floor((p+72)*self.config.density))
            if self.observer is not None:self.observer(l,query,kh[0].transpose(0,1).reshape(12,p+72,64,128),vh[0].transpose(0,1).reshape(12,p+72,64,128),summary,selection)
            out=SelectedAttention.run(query,kh[0].transpose(0,1).reshape(12,p+72,64,128),vh[0].transpose(0,1).reshape(12,p+72,64,128),summary,m.basis,selection).transpose(0,1)[None]
        if step==4:
            kg,vg,order=GroupingOps.group(m,k,v,self.config)
            store.commit(kg,vg,SummaryOps.build(m,kg,vg),order,start)
        with torch.no_grad():cache['global_end_index'].fill_(start+n);cache['local_end_index'].fill_(p*64+n)
        return module.o(out.flatten(2))
