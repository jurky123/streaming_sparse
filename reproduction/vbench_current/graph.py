"""Preallocated route + 512-action correction + compensation CUDA Graphs.

Stage page data once per layer/forward, then copy only each query batch and own
indices. Raw KV never enters this workspace: exact attention reads its resident
cache directly after the graph emits the final page LUT.
"""
import math
import torch
from ..review_v1.ops import Summary,RouteOps,haar
from .page import read
from .scores import candidate_pool,fused_grouped_scores as grouped_scores,apply_best
from .attention import compensate
from quota_mask import quota_mask

CACHED_LOGMASS=False

class BatchGraphs:
    def __init__(self,risk_function,phi,config,compile_route=False,compile_tail=False,action_kernel=False,tensor_math=False,projection_float=False,compact_reader=False,probes_only=False,bf16_qk=False,static_route=False,input_m16=False):
        assert not input_m16 or probes_only
        self.input_m16=input_m16;self.probes_only=probes_only;self.compact_reader=compact_reader;self.inline=False;self.projection_float=projection_float;self.risk_function=risk_function;self.phi=phi;self.config=config;self.entries={};self.calls=0;self.pool=torch.cuda.graph_pool_handle()
        if action_kernel:
            from .scores_action import fused_action_scores
            self.score_function=fused_action_scores
        else:self.score_function=grouped_scores
        self.native_route_qk = False
        self.projected_route_inputs = False
        self.probe_buffer = None
        self.scale_in_qk = False
        self.reuse_selected_ids = False
        self.fixed_page_sets = False
        self.radix_select = False
        self.anchor_policy = None
        self.partial_mass = False
        self.direct_risk_bound = False
        self.matched_probabilities = False
        self.dot=torch.bmm
        tail=compensate
        if tensor_math:
            from .tensor_bmm import tensor_bmm
            from .attention import compensate_tensor
            self.dot=tensor_bmm;tail=compensate_tensor
        if bf16_qk:
            from .tensor_bmm import bf16_inputs_bmm
            self.dot=bf16_inputs_bmm
        self.compensate=torch.compile(tail,fullgraph=True,options={"triton.cudagraphs":False}) if compile_tail else tail
        if compile_route:
            self.compute=torch.compile(self.compute,fullgraph=True,dynamic=False if static_route else None,options={"triton.cudagraphs":False})
    def select_probes(self,query,offsets):
        if not self.projected_route_inputs:
            return query.index_select(2,offsets)
        shape=(*query.shape[:2],offsets.numel(),query.shape[-1])
        buf=self.probe_buffer
        if buf is None or buf.shape!=shape or buf.device!=query.device or buf.dtype!=query.dtype:
            self.probe_buffer=query.new_empty(shape)
        from .route_probes import select
        return select(query,offsets,self.probe_buffer)

    @torch.no_grad()
    def get(self,q,summary,encoding,theta,basis,own,full_pset,heads=None,resident=False):
        h,t,n,d=q.shape;p=summary.mean.shape[1];selected_heads=heads if full_pset else None;key=(h,t,p,full_pset,q.device,q.dtype,n,0 if selected_heads is None else selected_heads.numel())
        key+=(self.n,)
        if resident:
            key+=('resident',summary.keys.data_ptr(),summary.stats.data_ptr())
        if key not in self.entries:
            entry=dict(q=q if q is self.probe_buffer else torch.empty_like(q),s=summary if resident else Summary(*(torch.empty_like(getattr(summary,name)) for name in ('keys','mean','detail','stats'))),
                       theta=torch.empty_like(theta),basis=basis if resident else torch.empty_like(basis),own=own if resident else torch.empty_like(own),
                       en={name:torch.empty_like(v) for name,v in encoding.items()} if full_pset else {},
                       offsets=torch.linspace(0,63,16,device=q.device).round().long())
            if selected_heads is not None:entry['active_heads']=selected_heads.clone()
            entry['haar']=haar(q).double();entry['haar_float']=haar(q)
            entry['resident']=resident
            if resident:entry['theta'].copy_(theta)
            else:self.stage(entry,summary,encoding,theta,basis)
            entry['q'].copy_(q)
            if not resident:entry['own'].copy_(own)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.compute(entry,full_pset)
            torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool):out=self.compute(entry,full_pset)
            graph.replay()
            if self.probes_only:
                entry.update(full_pset=full_pset,graph=graph,out=out,tail_graph=None,tail_out=())
                self.entries[key]=entry
                return entry
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.compensate(out[1],entry['s'],entry['basis'],out[2],entry['haar_float'])
            torch.cuda.current_stream().wait_stream(stream);tail_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(tail_graph,pool=self.pool):tail_out=self.compensate(out[1],entry['s'],entry['basis'],out[2],entry['haar_float'])
            entry.update(full_pset=full_pset,graph=graph,out=out,tail_graph=tail_graph,tail_out=tail_out);self.entries[key]=entry
        if selected_heads is not None:self.entries[key]['active_heads'].copy_(selected_heads)
        return self.entries[key]
    @torch.no_grad()
    def stage(self,e,s,en,theta,basis):
        fields=('keys','stats') if self.projected_route_inputs else ('keys','mean','detail','stats')
        for name in fields:
            getattr(e['s'],name).copy_(getattr(s,name))
        for name,value in e['en'].items():value.copy_(en[name])
        e['theta'].copy_(theta)
        if not self.projected_route_inputs:e['basis'].copy_(basis)
    def compute(self,e,full_pset):
        q=e['q'];h,t,n,d=q.shape;s=e['s'];p=s.mean.shape[1];en=e['en'];own=e['own'];theta=e['theta'];basis=e['basis']
        query=q.float();probes=query if self.input_m16 else query.index_select(2,e['offsets'])
        if (not self.anchor_policy and self.partial_mass and self.probes_only and self.input_m16 and self.reuse_selected_ids=='ties'
                and self.fixed_page_sets and not full_pset and p in (216,288,360,432)):
            from .risk_partial_mass import score
            from .fixed_page_select import select
            from .selected_id_order import tie_order
            if self.radix_select and self.partial_mass=='recheck':
                from .risk_partial_mass import select_scores
                risk,selected,mask=select_scores(q,s,theta,own,radix_select=True)
            else:
                risk=score(q,s,theta,own if self.partial_mass=='recheck' else None)
                selected,mask=select(risk,own)
            count=p//10
            ids=selected if count>32 else tie_order(risk,selected,count)
            # This probes-only replay exposes ids, mask, risk and counters.
            logits=q.new_empty((0,),dtype=torch.float32)
            return ids,logits,mask,risk,own.new_zeros(3)
        if self.scale_in_qk:
            from .tensor_bmm import scaled_route_qk
            z = scaled_route_qk(q.flatten(1, 2), s.keys.flatten(1, 2).transpose(1, 2)).reshape(h, t, 16, p, 4)
            logits = z
        elif self.native_route_qk:
            z = self.dot(q.flatten(1, 2), s.keys.flatten(1, 2).transpose(1, 2)).reshape(h, t, 16, p, 4) / math.sqrt(d)
            logits = z
        elif self.probes_only:
            z=self.dot(probes.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,16,p,4)/math.sqrt(d)
            logits=z
        else:
            logits=self.dot(query.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,n,p,4)/math.sqrt(d)
            z=logits.index_select(2,e['offsets'])
        if self.matched_probabilities and p in (216,288,360,432) and (not full_pset or 'active_heads' in e):
            from .risk_online import bounded_risk, matched_risk
            risk_fn=bounded_risk if self.direct_risk_bound and not self.anchor_policy else matched_risk
            risk=risk_fn(probes,z,s,theta)
        else:
            risk=self.risk_function(probes,z,s,theta)
        count=max(1,int(p*.1))
        fixed=self.fixed_page_sets and self.config.density==.1 and self.config.id_capacity==128 and p in (216,288,360,432)
        if self.anchor_policy:
            if p==72:
                # The first frame belongs to this chunk. Count it only once.
                page=torch.arange(p,device=q.device)
                ranked=risk.masked_fill(page<24,-torch.inf).scatter(-1,own[...,None],torch.inf)
                ids=ranked[...,24:].topk(self.n,dim=-1).indices+24
                mask=torch.zeros_like(risk,dtype=torch.bool).scatter_(-1,ids,True) | (page<24)
            else:
                mask=quota_mask(risk,own,self.current_slots,self.n)
            count=24+self.n
        elif fixed:
            from .fixed_page_select import select
            selected,mask=select(risk,own)
        else:
            sel=RouteOps.select(risk,own,self.config,check_valid=False)
            selected,mask=sel['ids'],sel['mask']
        if not self.anchor_policy and self.reuse_selected_ids and not full_pset and self.config.density==.1 and p in (216,288,360,432):
            stats=torch.zeros(3,device=q.device,dtype=torch.int64)
            if fixed and count>32:
                return selected,logits,mask,risk,stats
            if self.reuse_selected_ids=='ties':
                from .selected_id_order import tie_order
                return tie_order(risk,selected,count),logits,mask,risk,stats
            if self.reuse_selected_ids=='compact':
                from .selected_id_order import compact_order
                return compact_order(risk,selected,count),logits,mask,risk,stats
            return selected[...,:count].contiguous(),logits,mask,risk,stats
        if full_pset:
            full_mask=mask;full_risk=risk;full_own=own;full_h=h
            if 'active_heads' in e:
                heads=e['active_heads'];h=heads.numel()
                z=z.index_select(0,heads);probes=probes.index_select(0,heads)
                risk=risk.index_select(0,heads);mask=mask.index_select(0,heads);own=own.index_select(0,heads)
                en={k:v.index_select(0,heads) for k,v in en.items()}
            zz=z.double()
            if CACHED_LOGMASS:
                from .logmass_cached import normalized
                lm=normalized(z)
            else:
                lm=zz.logsumexp(-1);lm=lm-lm.logsumexp(-1,keepdim=True)
            b=h*t
            if self.anchor_policy:
                from quota_pool import pool
                r,a,legal=pool(risk,mask,own)
            else:
                r,a=candidate_pool(risk,mask,own)
            if self.compact_reader:
                from .reader_reduce import compact_scores
                scores=compact_scores(self.phi.query(probes),en,z,lm,mask,r,a)
            else:
                rp,beta=read(probes,en['co'],en['A'],self.phi)
                # haar is a constant GPU tensor supplied by the graph entry.
                if self.projection_float:
                    contrast=torch.einsum('htmpr,ar->htmpa',z.softmax(-1),e['haar_float'])
                    approx=en['mean'].float()[:,None,None]+torch.einsum('htmpa,hpak->htmpk',contrast,en['detail'].float())
                    center=approx-(lm.exp().float()[...,None]*approx).sum(-2,keepdim=True)
                else:
                    contrast=torch.einsum('htmpr,ar->htmpa',zz.softmax(-1),e['haar'])
                    approx=en['mean'][:,None,None]+torch.einsum('htmpa,hpak->htmpk',contrast,en['detail'])
                    center=approx-(lm.exp()[...,None]*approx).sum(-2,keepdim=True)
                scores=self.score_function(rp.reshape(b,16,p,16),beta.reshape(b,16,p),lm.reshape(b,16,p),center.reshape(b,16,p,16),mask.reshape(b,p),r.reshape(b,-1),a.reshape(b,-1))
            if self.anchor_policy:
                scores=scores.masked_fill(~legal.reshape_as(scores),-1e30)
            mask,gain,executed=apply_best(mask.reshape(b,p),r.reshape(b,-1),a.reshape(b,-1),scores);mask=mask.reshape(h,t,p)
            stats=torch.stack((executed.sum(),executed.new_full((),b,dtype=torch.int64),(~torch.isfinite(scores)).sum()))
            if 'active_heads' in e:mask=full_mask.index_copy(0,e['active_heads'],mask)
            risk=full_risk;own=full_own;h=full_h
        else:stats=torch.zeros(3,device=q.device,dtype=torch.int64)
        ids=risk.masked_fill(~mask,-torch.inf).scatter(-1,own[...,None],torch.inf).topk(count,dim=-1).indices.int()
        return ids,logits,mask,risk,stats
    @torch.no_grad()
    def replay(self,e,q,own):
        if e['q'] is not q:e['q'].copy_(q)
        if e['own'] is not own:e['own'].copy_(own)
        if self.inline or torch.cuda.is_current_stream_capturing():
            out=self.compute(e,e['full_pset']);e['inline_out']=out
        else:
            e['graph'].replay();out=e['out'];e['inline_out']=None
        self.calls+=1
        ids,logits,mask,risk,stats=out;return ids,mask,risk,stats
    @torch.no_grad()
    def tail(self,e):
        if self.inline or torch.cuda.is_current_stream_capturing():
            out=e['inline_out']
            return self.compensate(out[1],e['s'],e['basis'],out[2],e['haar_float'])
        e['tail_graph'].replay();return e['tail_out']
    def static_bytes(self):
        total=0
        seen_queries=set()
        for e in self.entries.values():
            tensors=([e['active_heads']] if 'active_heads' in e else [])+[e['theta'],e['basis'],e['own'],e['offsets'],e['haar'],e['haar_float'],*e['en'].values(),*(getattr(e['s'],n) for n in ('keys','mean','detail','stats')),*e['out'],*e['tail_out']]
            if e.get('resident',False):
                tensors=[e['theta'],e['offsets'],e['haar'],e['haar_float'],*e['out']]
            if id(e['q']) not in seen_queries:
                tensors.append(e['q'])
                seen_queries.add(id(e['q']))
            total+=sum(v.numel()*v.element_size() for v in tensors)
        return total
