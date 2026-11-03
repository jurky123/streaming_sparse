"""Explicit streaming lifecycle and quality-first implementation.

No CUDA graph capture or speed tuning. History summaries are immutable per
commit; only current summaries are recomputed. Attention uses selected raw KV
and unselected summaries. Dynamic counts use a slow reference fallback until
quality is established (device-loop kernel is a later implementation gate).
"""
import math,types
import torch
from .ops import Config,Summary,SummaryOps,RouteOps,haar
from ..response_hard import balanced
from ..cache_grouping import reorder

class PageStore:
    def __init__(self,cache,config):
        self.cache=cache;self.config=config;self.n=0;self.write=0;self.commits=0
        device=cache['k'].device
        self.summary=Summary(torch.empty(12,config.history_pages,4,128,device=device,dtype=torch.bfloat16),torch.empty(12,config.history_pages,128,device=device,dtype=torch.bfloat16),torch.empty(12,config.history_pages,3,8,device=device,dtype=torch.bfloat16),torch.empty(12,config.history_pages,4,12,device=device,dtype=torch.bfloat16))
        self.token_ids=torch.full((12,config.history_pages,64),-1,device=device,dtype=torch.int64)
    def reset(self):self.n=0;self.write=0;self.commits=0;self.token_ids.fill_(-1)
    @torch.no_grad()
    def commit(self,k,v,summary,order,start):
        c=self.config;slots=(torch.arange(c.current_pages,device=k.device)+self.write)%c.history_pages
        ids=(slots[:,None]*64+torch.arange(64,device=k.device)).flatten()
        self.cache['k'][0].index_copy_(0,ids,k.flatten(1,2).transpose(0,1))
        self.cache['v'][0].index_copy_(0,ids,v.flatten(1,2).transpose(0,1))
        for name in ('keys','mean','detail','stats'):getattr(self.summary,name).index_copy_(1,slots,getattr(summary,name))
        self.token_ids.index_copy_(1,slots,(order+start).reshape(12,72,64))
        self.write=(self.write+c.current_pages)%c.history_pages;self.n=min(c.history_pages,self.n+c.current_pages);self.commits+=1

class GroupingOps:
    @staticmethod
    @torch.no_grad()
    def group(model,k,v,config):
        # k already RoPE encoded. Each physical page owns all its representatives.
        k=k[0].transpose(0,1);v=v[0].transpose(0,1)
        f=model.group_features(k.float()[:,None],v.float()[:,None]).squeeze(1)
        order=balanced(f,[3,3,2,2,2]);fp=reorder(f,order)
        inner=balanced(fp.reshape(-1,64,32),[2,2]).reshape(12,72,64)
        order=order.reshape(12,72,64).gather(2,inner)[:,config.page_order(k.device)].flatten(1)
        return reorder(k,order).reshape(12,72,64,128),reorder(v,order).reshape(12,72,64,128),order

class SelectedAttention:
    @staticmethod
    def run(q,k,v,s,basis,selection):
        """Single tile live at a time. No exact attention on omitted pages.
        BF16-rounded inputs, FP32 reference QK/AV. Not final Tensor Core kernel.
        """
        if 'fixed_count' in selection:
            return SelectedAttention.fixed(q,k,v,s,basis,selection)
        h,n,d=q.shape;outputs=[];p=k.shape[1];basis=basis.bfloat16().float()
        for tile in range(n//64):
            query=q[:,tile*64:(tile+1)*64].bfloat16();mask=selection['mask'][:,tile]
            # Dynamic reference reads ONLY actual IDs. Deliberate CPU sync is
            # labelled, never presented as realized theoretical performance.
            counts=selection['counts'][:,tile]
            head_outputs=[]
            for head in range(h):
                count=int(counts[head]);ids=selection['ids'][head,tile,:count].long()
                kk=k[head].index_select(0,ids).flatten(0,1);vv=v[head].index_select(0,ids).flatten(0,1)
                z=(query[head].float()@kk.float().T)/math.sqrt(d)
                ze=z.logsumexp(-1);exact=z.softmax(-1)@vv.float()
                keys=s.keys[head].flatten(0,1)
                zs=(query[head].float()@keys.float().T)/math.sqrt(d)
                zs=zs.reshape(64,p,4).masked_fill(mask[head][None,:,None],-torch.inf)
                zt=zs.flatten(1).logsumexp(-1)+math.log(16)
                # Empty tail: use finite substitute before exp, zero weights.
                finite=torch.isfinite(zt);normalizer=torch.where(finite,zt-math.log(16),torch.zeros_like(zt))
                weights=(zs-normalizer[:,None,None]).exp()
                mean=weights.sum(-1)@s.mean[head].float()
                contrasts=torch.einsum('qpr,ar->qpa',weights,haar(q))
                codes=torch.einsum('qpa,pak->qk',contrasts,s.detail[head].float())
                tail=mean+codes@basis[head].T
                total=torch.logaddexp(ze,zt)
                head_outputs.append(exact*(ze-total).exp()[:,None]+tail*(zt-total).exp()[:,None])
            outputs.append(torch.stack(head_outputs))
        return torch.cat(outputs,1).to(q.dtype)

    @staticmethod
    def fixed(q,k,v,s,basis,selection):
        """Ordinary batched PyTorch quality path, same FP32 numerical contract.
        One tile live; heads batched. No kernel tuning or speed claim.
        """
        h,n,d=q.shape;p=k.shape[1];count=selection['fixed_count'];outputs=[]
        keys=k.flatten(1,2);values=v.flatten(1,2);reps=s.keys.flatten(1,2).float()
        basis=basis.bfloat16().float();offsets=torch.arange(64,device=q.device)
        for tile in range(n//64):
            query=q[:,tile*64:(tile+1)*64].bfloat16().float()
            ids=selection['ids'][:,tile,:count].long()
            tokens=(ids[:,:,None]*64+offsets).flatten(1)
            index=tokens[:,:,None].expand(-1,-1,d)
            kk=keys.gather(1,index).float();vv=values.gather(1,index).float()
            z=torch.bmm(query,kk.transpose(1,2))/math.sqrt(d)
            ze=z.logsumexp(-1);exact=torch.bmm(z.softmax(-1),vv)
            zs=torch.bmm(query,reps.transpose(1,2)).reshape(h,64,p,4)/math.sqrt(d)
            zs=zs.masked_fill(selection['mask'][:,tile,None,:,None],-torch.inf)
            zt=zs.flatten(2).logsumexp(-1)+math.log(16)
            normalizer=torch.where(torch.isfinite(zt),zt-math.log(16),torch.zeros_like(zt))
            weights=(zs-normalizer[:,:,None,None]).exp()
            mean=torch.bmm(weights.sum(-1),s.mean.float())
            contrasts=torch.einsum('hqpr,ar->hqpa',weights,haar(q))
            code=torch.bmm(contrasts.flatten(2),s.detail.float().flatten(1,2))
            tail=mean+torch.bmm(code,basis.transpose(1,2))
            total=torch.logaddexp(ze,zt)
            outputs.append(exact*(ze-total).exp()[...,None]+tail*(zt-total).exp()[...,None])
        return torch.cat(outputs,1).to(q.dtype)

class StreamingRuntime:
    def __init__(self,engine,model,config=Config(),thresholds=None):
        self.engine=engine;self.model=model;self.config=config;self.thresholds=thresholds;self.stores={};self.sparse=True
        self.original=[b.self_attn.forward for b in engine.wrapper.model.blocks];self.trace=[]
        self.original_blocks=[b.forward for b in engine.wrapper.model.blocks];self.training_event=False
        for l,b in enumerate(engine.wrapper.model.blocks):
            cap=360 if l==0 else config.history_pages;tokens=(cap+72)*64;c=engine.cache[l]
            if c['k'].shape[1]!=tokens:
                c['k']=torch.empty(1,tokens,12,128,device='cuda',dtype=torch.bfloat16);c['v']=torch.empty_like(c['k'])
            b.self_attn.max_attention_size=tokens
            if l==0:continue
            self.stores[l]=PageStore(c,config)
            def forward(module,x,seq_lens,grid_sizes,freqs,block_mask,kv_cache=None,current_start=0,cache_start=None,index=l):
                return self.forward(index,module,x,grid_sizes,freqs,kv_cache,current_start)
            b.self_attn.forward=types.MethodType(forward,b.self_attn)
        self.reset()
    def reset(self):
        self.engine.reset();self.trace=[]
        for s in self.stores.values():s.reset()
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
            out=SelectedAttention.run(query,kh[0].transpose(0,1).reshape(12,p+72,64,128),vh[0].transpose(0,1).reshape(12,p+72,64,128),summary,m.basis,selection).transpose(0,1)[None]
        if step==4:
            kg,vg,order=GroupingOps.group(m,k,v,self.config)
            store.commit(kg,vg,SummaryOps.build(m,kg,vg),order,start)
        with torch.no_grad():cache['global_end_index'].fill_(start+n);cache['local_end_index'].fill_(p*64+n)
        return module.o(out.flatten(2))
    def finish(self):
        raw=sum(c[n].numel()*c[n].element_size() for c in self.engine.cache for n in ('k','v'))
        summaries=sum(s.summary.bytes() for s in self.stores.values())
        identities=sum(s.token_ids.numel()*s.token_ids.element_size() for s in self.stores.values())
        params=sum(p.numel()*p.element_size() for p in self.model.parameters())
        total=raw+summaries+identities+params
        return dict(status='quality_reference_not_speed_accepted',raw_bytes=raw,summary_bytes=summaries,token_identity_bytes=identities,parameter_bytes=params,persistent_bytes=total,limit_bytes=5112856576,within_limit=total<=5112856576,history_pages=self.config.history_pages,clean='dense; full commit; no terminal trimming',dynamic_execution='per-head CPU count reference; no graph claim')
    @torch.no_grad()
    def refresh_summaries(self):
        # After a training update, preserve physical groups/raw KV; recompute only
        # their summaries. Calibration labels and thresholds must be regenerated.
        for l,store in self.stores.items():
            p=store.n
            if not p:continue
            k=store.cache['k'][0,:p*64].transpose(0,1).reshape(12,p,64,128)
            v=store.cache['v'][0,:p*64].transpose(0,1).reshape_as(k)
            summary=SummaryOps.build(self.model[l],k,v)
            for name in ('keys','mean','detail','stats'):getattr(store.summary,name)[:,:p].copy_(getattr(summary,name))
    def close(self):
        for b,f in zip(self.engine.wrapper.model.blocks,self.original):b.self_attn.forward=f
        for b,f in zip(self.engine.wrapper.model.blocks,self.original_blocks):b.forward=f
