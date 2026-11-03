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

class DownstreamRouteRuntime(StreamingRuntime):
    def __init__(self,engine,model,variant,centers,**kwargs):
        assert variant in VARIANTS
        self.variant=variant;self.centers=centers;self.intervention=None;self.observation=None
        super().__init__(engine,model,**kwargs)
    def group(self,l,k,v):
        if self.variant!='no_grouping':return GroupingOps.group(self.model[l],k,v,self.config)
        order=torch.arange(4608,device=k.device).reshape(72,64)[self.config.page_order(k.device)].flatten()[None].expand(12,-1)
        idx=order[:,:,None].expand(-1,-1,128)
        kr=k[0].transpose(0,1).gather(1,idx).reshape(12,72,64,128)
        vr=v[0].transpose(0,1).gather(1,idx).reshape_as(kr)
        return kr,vr,order
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
            scores=[]; feature_tiles=[]
            with torch.no_grad():
                for tile in range(0,72,8):
                    probes=query[:,tile*64:(tile+8)*64].reshape(12,8,64,128).index_select(2,offsets).flatten(1,2)
                    f,_=RouteOps.features(probes,summary)
                    if self.intervention is not None and l==self.intervention[0] and step==0:
                        feature_tiles.append(f.reshape(12,8,self.config.probes,p+72,4).mean(2))
                    scores.append(mass_scores(probes,summary,8,self.config.probes) if self.variant=='mass_route' else (f*m.theta[:,None,None]).sum(-1).reshape(12,8,self.config.probes,p+72).mean(2))
                scores=torch.cat(scores,1);own=(torch.arange(72,device=q.device)+p)[None].expand(12,-1)
                threshold=None if self.thresholds is None else self.thresholds[l,:,step][:,None,None]
                selection=RouteOps.select(scores,own,self.config,threshold)
                if threshold is None:selection['fixed_count']=max(1,math.floor((p+72)*self.config.density))
                if self.intervention is not None and l==self.intervention[0] and step==0:
                    _,head,offset,apply=self.intervention
                    features=torch.cat(feature_tiles,1)
                    selection,self.observation=swap_boundary(selection,scores,own,features,head,offset,apply)

            # Transform only after routing: remove compensation information
            # without silently changing the router's input statistics.
            summary=change_summary(summary,m.basis,self.centers.get(l),self.variant)
            out=SelectedAttention.run(query,kh[0].transpose(0,1).reshape(12,p+72,64,128),vh[0].transpose(0,1).reshape(12,p+72,64,128),summary,m.basis,selection).transpose(0,1)[None]
        if step==4:
            kg,vg,order=self.group(l,k,v)
            store.commit(kg,vg,SummaryOps.build(m,kg,vg),order,start)
        with torch.no_grad():cache['global_end_index'].fill_(start+n);cache['local_end_index'].fill_(p*64+n)
        return module.o(out.flatten(2))


def swap_boundary(selection,scores,own,features,head,offset,apply):
    """Swap one unprotected boundary page per tile in one head, fixed count."""
    count=selection['fixed_count'];mask=selection['mask'][head]
    chosen=scores[head].masked_fill(~mask,torch.inf).clone()
    chosen.scatter_(-1,own[head,:,None],torch.inf)
    drop=chosen.argmin(-1)
    add=scores[head].masked_fill(mask,-torch.inf).topk(offset+1,dim=-1).indices[:,offset]
    tiles=torch.arange(scores.shape[1],device=scores.device)
    delta=(features[head,tiles,add]-features[head,tiles,drop]).mean(0)
    assert torch.isfinite(delta).all() and (drop!=own[head]).all()
    observation=dict(delta=delta.cpu().tolist(),drop=drop.cpu().tolist(),add=add.cpu().tolist())
    if apply:
        selection={k:v.clone() if torch.is_tensor(v) else v for k,v in selection.items()}
        ids=selection['ids'][head,:,:count]
        assert (ids==drop[:,None]).sum().item()==scores.shape[1]
        ids.copy_(torch.where(ids==drop[:,None],add[:,None],ids))
        selection['mask'][head,tiles,drop]=False;selection['mask'][head,tiles,add]=True
        assert (selection['mask'].sum(-1)==count).all()
        assert selection['mask'].gather(-1,own[...,None]).all()
    return selection,observation
