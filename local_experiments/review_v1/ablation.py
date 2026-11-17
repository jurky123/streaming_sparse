"""Frozen-weight causal ablations; no changes to the accepted runtime files."""
import math,torch
from .ops import Summary,SummaryOps,RouteOps
from .runtime import StreamingRuntime,GroupingOps,SelectedAttention

VARIANTS=('full','projected_mean','no_detail','mass_route','no_grouping','no_new_training','no_downstream')

def change_summary(s,basis,center,variant):
    if variant=='no_detail':return Summary(s.keys,s.mean,torch.zeros_like(s.detail),s.stats)
    if variant=='projected_mean':
        u=basis.bfloat16().float()
        code=torch.bmm(s.mean.float()-center[:,None].float(),u).bfloat16().float()
        mean=center[:,None].float()+torch.bmm(code,u.transpose(1,2))
        return Summary(s.keys,mean,s.detail,s.stats)
    return s

def mass_scores(q,s,tiles,probes=16):
    h,n,d=q.shape;p=s.keys.shape[1]
    z=torch.einsum('hqd,hprd->hqpr',q.bfloat16().float(),s.keys.float())/math.sqrt(d)
    return z.flatten(2).softmax(-1).reshape(h,tiles,probes,p,4).sum(-1).mean(2)

class AblationRuntime(StreamingRuntime):
    def __init__(self,engine,model,variant,centers,**kwargs):
        assert variant in VARIANTS
        self.variant=variant;self.centers=centers
        super().__init__(engine,model,**kwargs)
    def group(self,l,k,v):
        if self.variant!='no_grouping':return GroupingOps.group(self.model[l],k,v,self.config)
        order=torch.arange(4608,device=k.device).reshape(72,64)[self.config.page_order(k.device)].flatten()[None].expand(12,-1)
        idx=order[:,:,None].expand(-1,-1,128)
        kr=k[0].transpose(0,1).gather(1,idx).reshape(12,72,64,128)
        vr=v[0].transpose(0,1).gather(1,idx).reshape_as(kr)
        return kr,vr,order
    def route_scores(self,probes,summary,model):
        f,_=RouteOps.features(probes,summary)
        return mass_scores(probes,summary,8,self.config.probes) if self.variant=='mass_route' else (f*model.theta[:,None,None]).sum(-1).reshape(12,8,self.config.probes,summary.keys.shape[1]).mean(2)
    def build_summary(self,model,k,v,cheap=False):
        return SummaryOps.build(model,k,v,cheap)
    def query_route_scores(self,query,summary,model):
        offsets=torch.linspace(0,63,self.config.probes,device=query.device).round().long()
        scores=[]
        for tile in range(0,72,8):
            probes=query[:,tile*64:(tile+8)*64].reshape(12,8,64,128).index_select(2,offsets).flatten(1,2)
            scores.append(self.route_scores(probes,summary,model))
        return torch.cat(scores,1)
    def apply_rotary(self,q,k,grid_sizes,freqs,start_frame):
        from wan.modules.causal_model import causal_rope_apply
        return causal_rope_apply(q,grid_sizes,freqs,start_frame).contiguous(),causal_rope_apply(k,grid_sizes,freqs,start_frame).contiguous()
    def history_and_current(self,cache,k,v,p):
        return torch.cat((cache['k'][:,:p*64].detach(),k),1),torch.cat((cache['v'][:,:p*64].detach(),v),1)
    def select_pages(self,scores,own,threshold):
        return RouteOps.select(scores,own,self.config,threshold)
    def selected_attention(self,*args):
        return SelectedAttention.run(*args)
    def forward(self,l,module,x,grid_sizes,freqs,cache,start):
        from wan.modules.causal_model import causal_rope_apply
        store=self.stores[l];m=self.model[l];p=store.n;b,n,_=x.shape
        assert (b,n)==(1,4608)
        q=module.norm_q(module.q(x)).view(b,n,12,128);k=module.norm_k(module.k(x)).view_as(q);v=module.v(x).view_as(q)
        q,k=self.apply_rotary(q,k,grid_sizes,freqs,start//1536)
        q=q.to(v.dtype);k=k.to(v.dtype)
        # History detached, current gradients retained; no implicit crosschunk BPTT.
        kh,vh=self.history_and_current(cache,k,v,p)
        step=self.engine.current['step']
        if not self.sparse or step==4 or p<=72:out=self.engine.dense(q,kh,vh)
        else:
            current=self.build_summary(m,k[0].transpose(0,1).reshape(12,72,64,128),v[0].transpose(0,1).reshape(12,72,64,128),cheap=True)
            history=store.summary.slice(0,p)
            if torch.is_grad_enabled():
                history=self.build_summary(m,kh[0,:p*64].transpose(0,1).reshape(12,p,64,128),vh[0,:p*64].transpose(0,1).reshape(12,p,64,128))
            summary=Summary(*(torch.cat((getattr(history,name),getattr(current,name)),1) for name in ('keys','mean','detail','stats')))
            query=q[0].transpose(0,1)
            with torch.no_grad():
                scores=self.query_route_scores(query,summary,m)
                own=(torch.arange(72,device=q.device)+p)[None].expand(12,-1)
                threshold=None if self.thresholds is None else self.thresholds[l,:,step][:,None,None]
                selection=self.select_pages(scores,own,threshold)
                if threshold is None:selection['fixed_count']=max(1,math.floor((p+72)*self.config.density))
            # Transform only after routing: remove compensation information
            # without silently changing the router's input statistics.
            summary=change_summary(summary,m.basis,self.centers[l],self.variant)
            out=self.selected_attention(query,kh[0].transpose(0,1).reshape(12,p+72,64,128),vh[0].transpose(0,1).reshape(12,p+72,64,128),summary,m.basis,selection).transpose(0,1)[None]
        if step==4:
            kg,vg,order=self.group(l,k,v)
            store.commit(kg,vg,self.build_summary(m,kg,vg),order,start)
        with torch.no_grad():cache['global_end_index'].fill_(start+n);cache['local_end_index'].fill_(p*64+n)
        return module.o(out.flatten(2))
