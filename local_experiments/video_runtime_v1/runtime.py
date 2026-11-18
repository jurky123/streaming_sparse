"""Resident KV + commit-time page encodings, shared GPU route/readout/compensation."""
import contextlib,math,time
import torch
from ..review_v1.ops import Config,Summary,RouteOps,haar
from ..review_v1.runtime import StreamingRuntime
from ..review_v1.graph_summary import GraphSummary
from ..review_v1.cached_ops import GraphGrouping
from .attention import Workspace,exact,compensate,merge
from .page import encode,read
from .scores import fused_grouped_scores as grouped_scores,candidate_pool,apply_best

class Timings:
    def __init__(self,enabled=False):self.enabled=enabled;self.events=[]
    @contextlib.contextmanager
    def stage(self,name):
        if not self.enabled:yield;return
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        try:yield
        finally:b.record();self.events.append((name,a,b))
    def finish(self,clear=False):
        torch.cuda.synchronize();out={}
        for name,a,b in self.events:
            row=out.setdefault(name,dict(milliseconds=0.,calls=0));row['milliseconds']+=a.elapsed_time(b);row['calls']+=1
        if clear:self.events.clear()
        return out

def risk_from_logits(probes,z,summary,theta):
    """Same four Risk equations; use the M16 rows of already computed Q64 logits."""
    h,t,m,d=probes.shape;q=probes.bfloat16().float();st=summary.stats.float()
    prob=z.flatten(-2).softmax(-1).reshape_as(z);alpha=prob.sum(-1)
    q2=q.square().reshape(h,t,m,8,16).sum(-1)/d;qn=q.norm(dim=-1)/math.sqrt(d)
    u=torch.einsum('htmg,hprg->htmpr',q2,st[...,:8]).clamp_min(0)
    beta=qn[...,None,None]*st[:,None,None,:,:,8]
    bp=st[:,:,1,11];c=bp+bp.amax(-1,keepdim=True)
    x0=alpha*st[:,None,None,:,0,11]+((prob-alpha[...,None]/4).abs()*st[:,None,None,:,:,9]).sum(-1)
    x1=(prob*u.sqrt()*st[:,None,None,:,:,10]).sum(-1)
    x2=c[:,None,None]*(prob*beta).sum(-1)
    x3=c[:,None,None]*(prob*(u+beta.square())).sum(-1)
    f=torch.stack((x0,x1,x2,x3),-1)
    return (f*theta[:,None,None,None]).sum(-1).mean(2)

class VideoRuntime(StreamingRuntime):
    def __init__(self,engine,model,reader,phi,B,*,full_pset=False,tile_batch=2,profile=False,config=Config(),validate=False,graph_batches=True,compile_route=False,compile_tail=False,action_kernel=False,tensor_math=False,projection_float=False,compact_reader=False,stream_attention=False,stream_tail_mode=4,stream_warps=4,bf16_qk=False,split_summary=False,fused_risk=False,static_route=False,legacy_components=None,projected_risk=False,cached_logmass=False,coeff_bf16=False,pset_steps='all',route_m16=False):
        if tile_batch<1 or 72%tile_batch:raise ValueError('tile_batch must divide 72')
        if config.probes!=16 or config.density!=.1:raise ValueError('Frozen configuration requires M16 / 10%')
        # The deployed default uses the prior direct-LUT exact and full-mean tail.
        # An explicit string (including "") preserves historical controls.
        if legacy_components is None:legacy_components="attention"
        self.legacy_components=set(legacy_components.split(",")) if legacy_components else set()
        self.legacy_attention="attention" in self.legacy_components
        self.stream_tail_mode=stream_tail_mode;self.stream_warps=stream_warps;self.bf16_qk=bf16_qk;self.stream_attention=stream_attention;self.value_slabs={};self.reader=reader;self.phi=phi;self.B=B.double() if B is not None else None;self.full_pset=full_pset;self.pset_schedule=None;self.pset_layers=None;self.pset_step_ids=None;self.pset_head_ids=None;self.pset_steps=pset_steps;self.route_m16=route_m16;self.tile_batch=tile_batch
        self.workspace=Workspace();self.timings=Timings(profile);self.summary_graph=GraphSummary();self.group_graph=GraphGrouping(config)
        self.split_summary=split_summary
        if split_summary:
            from ..review_v1.split_summary import build
            self.summary_graph=build
        if "summary" in self.legacy_components:
            from functools import partial
            from ..review_v1.split_summary import build
            self.summary_graph=partial(build,three_term_features=True)
        self.graph_batches=graph_batches
        from .pipeline_graph import BatchGraphs
        self.static_route=static_route;self.fused_risk=fused_risk
        risk_fn=risk_from_logits
        if fused_risk:
            from .risk_fused import risk
            risk_fn=risk
        if projected_risk:
            from .risk_projected import risk
            risk_fn=risk
        self.projected_risk=projected_risk
        from . import pipeline_graph
        pipeline_graph.CACHED_LOGMASS=cached_logmass
        self.cached_logmass=cached_logmass
        self.coeff_bf16=coeff_bf16
        if "risk" in self.legacy_components:
            from .legacy_components import risk
            risk_fn=risk
        self.batch_graph=BatchGraphs(risk_fn,phi,config,compile_route=compile_route,compile_tail=compile_tail,action_kernel=action_kernel,tensor_math=tensor_math,projection_float=projection_float,compact_reader=compact_reader,probes_only=stream_attention,bf16_qk=bf16_qk,static_route=static_route,input_m16=route_m16)
        self.tensor_math=tensor_math;self.compile_route=compile_route;self.compile_tail=compile_tail;self.action_kernel=action_kernel
        self.validate=validate;self.validation=[];self.checked=set();self.slabs={};self.encodings={};self.projected_basis={}
        super().__init__(engine,model,config)
        h=12;p=config.history_pages+72;device=self.model[1].basis.device
        self.haar_basis=haar(self.model[1].basis);self.offsets=torch.linspace(0,63,16,device=device).round().long()
        for l,store in self.stores.items():
            old=store.summary
            slab=Summary(*(torch.empty((h,p,*getattr(old,n).shape[2:]),device=device,dtype=torch.bfloat16) for n in ('keys','mean','detail','stats')))
            self.slabs[l]=slab;store.summary=slab.slice(0,config.history_pages)
            if self.B is not None:self.encodings[l]=dict(co=torch.empty(h,p,5,17,device=device,dtype=torch.bfloat16 if coeff_bf16 else torch.float32),A=torch.empty(h,p,8,4,device=device,dtype=torch.bfloat16 if coeff_bf16 else torch.float32),mean=torch.empty(h,p,16,device=device,dtype=torch.float64),detail=torch.empty(h,p,3,16,device=device,dtype=torch.float64))
            if self.B is not None:self.projected_basis[l]=self.model[l].basis.bfloat16().double().transpose(1,2)@self.B
            if stream_attention and not self.legacy_attention:self.value_slabs[l]=torch.empty(12,p,4,128,device=device,dtype=torch.float32)
        self.counters=torch.zeros(4,device=device,dtype=torch.int64)
        self.resident_own=None
        self.commit_stream=None
        self.clean_group_batch=0
        self.pending_clean=[]
        self.last_sparse_layer=max(self.stores)

    def set_pset_scope(self,layers=None,heads=None,steps=None):
        """Independent deployment scope; change only between videos, then reset."""
        self.pset_schedule=None
        self.pset_layers=None if layers is None else frozenset(layers)
        self.pset_step_ids=None if steps is None else frozenset(steps)
        self.pset_head_ids=None if heads is None or len(heads)==12 else torch.tensor(heads,device=self.offsets.device,dtype=torch.long)
    def set_pset_schedule(self,schedule):
        """Map (layer,noisy step) to selected heads; omitted cells stay on base path."""
        self.set_pset_scope(layers=sorted({int(k.split(':')[0]) for k,v in schedule.items() if v}))
        self.pset_schedule={tuple(map(int,k.split(':'))):torch.tensor(v,device=self.offsets.device,dtype=torch.long) for k,v in schedule.items() if v}
    def close(self):
        if getattr(self,'commit_stream',None) is not None:
            torch.cuda.current_stream().wait_stream(self.commit_stream)
        if hasattr(self,'repeated_context'):self.repeated_context.close()
        if hasattr(self,'normalized_rotary'):self.normalized_rotary.close()
        if hasattr(self,'rounded_ffn'):self.rounded_ffn.close()
        if hasattr(self,'native_rope'):self.native_rope.close()
        if hasattr(self,'rms_dataflow'):self.rms_dataflow.close()
        if hasattr(self,'pointwise_blocks'):self.pointwise_blocks.close()
        super().close()
    def correction_scope(self,l,step):
        """Current correction branch, shared by eager execution and block capture."""
        layer_enabled=self.full_pset and (self.pset_layers is None or l in self.pset_layers);correct=layer_enabled and (step in self.pset_step_ids if self.pset_step_ids is not None else (self.pset_steps=='all' or step==3))
        heads=self.pset_head_ids
        if self.pset_schedule is not None:
            heads=self.pset_schedule.get((l,step));correct=layer_enabled and heads is not None
            if heads is not None and heads.numel()==12:heads=None
        return correct,heads
    def reset(self):
        if getattr(self,'commit_stream',None) is not None:
            torch.cuda.current_stream().wait_stream(self.commit_stream)
        if hasattr(self,"pending_clean"):self.pending_clean.clear()
        super().reset()
        if hasattr(self,'counters'):self.counters.zero_()
        if hasattr(self,'timings'):self.timings.events.clear()
    def encode(self,l,s):
        self.counters[3]+=s.mean.shape[0]*s.mean.shape[1]
        co,A,mean=encode(s,self.reader,self.phi,self.B,coeff_bf16=self.coeff_bf16)
        detail=torch.matmul(s.detail.double(),self.projected_basis[l][:,None])
        return dict(co=co,A=A,mean=mean,detail=detail)
    def write_encoding(self,l,value,*,start=None,slots=None):
        for name,x in value.items():
            if slots is None:self.encodings[l][name][:,start:start+x.shape[1]].copy_(x)
            else:self.encodings[l][name].index_copy_(1,slots,x)
    @torch.no_grad()
    def forward(self,l,module,x,grid_sizes,freqs,cache,start):
        from wan.modules.causal_model import causal_rope_apply
        store=self.stores[l];model=self.model[l];p=store.n;b,n,_=x.shape
        assert (b,n)==(1,4608) and not torch.is_grad_enabled()
        project_rotary=getattr(module,'project_normalized_rotary',None)
        end=p*64+n
        key_window=cache['k'][:,p*64:end]
        value_window=cache['v'][:,p*64:end]
        project_cached=getattr(module,'project_cached_rotary',None)
        cached=(project_rotary is not None and project_cached is not None
                and key_window.is_contiguous() and value_window.is_contiguous()
                and key_window.dtype==value_window.dtype==x.dtype==torch.bfloat16)
        if cached:
            q,k,v=project_cached(module,x,grid_sizes,freqs,start//1536,key_window,value_window)
        elif project_rotary is not None:
            q,k=project_rotary(module,x,grid_sizes,freqs,start//1536)
            v=module.v(x).view_as(q)
        else:
            q=module.norm_q(module.q(x)).view(b,n,12,128);k=module.norm_k(module.k(x)).view_as(q)
            v=module.v(x).view_as(q)
            if "rope" in self.legacy_components:
                from ..review_v1.fused_rope import apply
                q,k=apply(q,k,grid_sizes,freqs,start//1536)
            else:
                q=causal_rope_apply(q,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
                k=causal_rope_apply(k,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        if not cached:
            key_window.copy_(k)
            value_window.copy_(v)
        kh=cache['k'][:,:end];vh=cache['v'][:,:end];step=self.engine.current['step'];correct,heads=self.correction_scope(l,step)
        layer_enabled=self.full_pset and (self.pset_layers is None or l in self.pset_layers)
        if not self.sparse or step==4 or p<=72:
            with self.timings.stage('attention'):out=self.engine.dense(q,kh,vh)
        else:
            with self.timings.stage('summary_and_encoding'):
                inline=(getattr(self,"inline_summary",False) or (torch.cuda.is_current_stream_capturing() and not self.split_summary))
                slab=self.slabs[l]
                kg=k[0].transpose(0,1).reshape(12,72,64,128)
                vg=v[0].transpose(0,1).reshape(12,72,64,128)
                if getattr(self,'summary_window',False) and not inline:
                    current=self.summary_graph(model,kg,vg,True,out=slab.slice(p,p+72))
                else:
                    if inline:
                        from ..review_v1.ops import SummaryOps
                        current=SummaryOps.build(model,kg,vg,True,haar_basis=self.haar_basis)
                    else:
                        current=self.summary_graph(model,kg,vg,True)
                    for name in ('keys','mean','detail','stats'):
                        getattr(slab,name)[:,p:p+72].copy_(getattr(current,name))
                if correct:self.write_encoding(l,self.encode(l,current),start=p)
                if self.stream_attention and not self.legacy_attention:
                    from .stream_attention import full_values
                    self.value_slabs[l][:,p:p+72].copy_(full_values(current,model.basis,self.haar_basis))
            s=self.slabs[l].slice(0,p+72);en={name:value[:,:p+72] for name,value in self.encodings.get(l,{}).items()}
            direct=self.tile_batch==72 and getattr(self,'direct_output',False)
            if not direct:result=self.workspace.tensor('result',q.shape,q.device,q.dtype)
            for first in range(0,72,self.tile_batch):
                t=self.tile_batch;qq=q[:,first*64:(first+t)*64]
                route_q=qq[0].transpose(0,1).reshape(12,t,64,128)
                if self.route_m16:
                    route_q = self.batch_graph.select_probes(route_q, self.offsets)
                if not self.batch_graph.native_route_qk:
                    route_q = route_q.float()
                with self.timings.stage('route_and_correction'):
                    resident=(getattr(self,'resident_route',False) and self.graph_batches and t==72
                              and not correct and self.batch_graph.partial_mass=='recheck' and s.keys.is_contiguous()
                              and s.stats.is_contiguous() and p+72==432)
                    if resident:
                        if self.resident_own is None:
                            self.resident_own=torch.arange(p,p+t,device=q.device)[None].expand(12,-1).contiguous()
                        own=self.resident_own
                    else:
                        own=(torch.arange(first,first+t,device=q.device)+p)[None].expand(12,-1)
                    if self.graph_batches:
                        entry=self.batch_graph.get(route_q,s,en,model.theta,model.basis,own,correct,heads=heads,resident=resident)
                        if resident:entry['theta'].copy_(model.theta)
                        elif first==0:self.batch_graph.stage(entry,s,en,model.theta,model.basis)
                        ids,mask,risk,stats=self.batch_graph.replay(entry,route_q,own)
                        if not resident:self.counters[:3]+=stats
                        count=ids.shape[-1]
                    else:
                        query=qq[0].transpose(0,1).float()
                        logits=torch.bmm(query,s.keys.float().flatten(1,2).transpose(1,2)).reshape(12,t,64,p+72,4)/math.sqrt(128)
                        probes=query.reshape(12,t,64,128).index_select(2,self.offsets)
                        z=logits.index_select(2,self.offsets);risk=risk_from_logits(probes,z,s,model.theta)
                        own=(torch.arange(first,first+t,device=q.device)+p)[None].expand(12,-1)
                        sel=RouteOps.select(risk,own,self.config,check_valid=False);mask=sel['mask'];count=max(1,math.floor((p+72)*.1))
                        if correct:
                            rp,beta=read(probes,en['co'],en['A'],self.phi)
                            zz=z.double();lm=zz.logsumexp(-1);lm=lm-lm.logsumexp(-1,keepdim=True)
                            contrast=torch.einsum('htmpr,ar->htmpa',zz.softmax(-1),haar(zz).double())
                            approx=en['mean'][:,None,None]+torch.einsum('htmpa,hpak->htmpk',contrast,en['detail'])
                            center=approx-(lm.exp()[...,None]*approx).sum(-2,keepdim=True)
                            r,a=candidate_pool(risk,mask,own);batch=12*t
                            score=grouped_scores(rp.reshape(batch,16,p+72,16),beta.reshape(batch,16,p+72),lm.reshape(batch,16,p+72),center.reshape(batch,16,p+72,16),mask.reshape(batch,p+72),r.reshape(batch,-1),a.reshape(batch,-1))
                            changed,gain,executed=apply_best(mask.reshape(batch,p+72),r.reshape(batch,-1),a.reshape(batch,-1),score)
                            mask=changed.reshape(12,t,p+72)
                            self.counters[0]+=executed.sum();self.counters[1]+=batch;self.counters[2]+=(~torch.isfinite(score)).sum()
                        ranked=risk.masked_fill(~mask,-torch.inf).scatter(-1,own[...,None],torch.inf)
                        ids=ranked.topk(count,dim=-1).indices.int()
                with self.timings.stage('attention'):
                    if self.legacy_attention:
                        from .legacy_components import attention
                        mixed=attention(qq,kh,vh,s,model.basis,ids,self.workspace)
                    elif self.stream_attention:
                        from .stream_attention import mixed_attention
                        mixed=mixed_attention(qq,kh,vh,s,self.value_slabs[l][:,:p+72],ids,mask,self.workspace,tail_mode=self.stream_tail_mode,warps=self.stream_warps)
                    else:
                        oe,le=exact(qq,kh,vh,ids,self.workspace)
                        oa,la=self.batch_graph.tail(entry) if self.graph_batches else compensate(logits,s,model.basis,mask)
                        mixed=merge(oe,le,oa,la)
                    if direct:result=mixed
                    else:result[:,first*64:(first+t)*64].copy_(mixed)
                key=(l,self.engine.current['chunk'],step)
                if self.validate and first==0 and key not in self.checked:
                    from ..review_v1.runtime import SelectedAttention
                    from ..review_v1.ops import RouteOps as ReferenceRoute
                    query=qq[0].transpose(0,1).float()
                    probes=query.reshape(12,t,64,128).index_select(2,self.offsets)
                    f,_=ReferenceRoute.features(probes.flatten(1,2),s)
                    refscore=(f*model.theta[:,None,None]).sum(-1).reshape(12,t,16,p+72).mean(2)
                    kr=kh[0].transpose(0,1).reshape(12,p+72,64,128);vr=vh[0].transpose(0,1).reshape_as(kr)
                    ref=SelectedAttention.fixed(query.bfloat16(),kr,vr,s,model.basis,dict(ids=ids,mask=mask,fixed_count=count))
                    delta=mixed[0].transpose(0,1).float()-ref.float();nmse=float(delta.square().sum()/ref.float().square().sum().clamp_min(1e-12))
                    err=float(delta.abs().max());riskerr=float((risk-refscore).abs().max()/refscore.abs().max().clamp_min(1e-12))
                    assert nmse<2e-5 and err<=.05 and riskerr<5e-5,(key,nmse,err,riskerr)
                    self.validation.append(dict(key=key,attention_nmse=nmse,attention_max_abs=err,risk_relative_max=riskerr));self.checked.add(key)
            out=result
        if step==4:
            with self.timings.stage('clean_commit'):
                if self.clean_group_batch:
                    self.pending_clean.append((l,model,k,v,start,layer_enabled))
                    if len(self.pending_clean)==self.clean_group_batch or l==self.last_sparse_layer:
                        jobs=self.pending_clean
                        if self.commit_stream is None:
                            self.commit_jobs(jobs)
                        else:
                            main=torch.cuda.current_stream()
                            self.commit_stream.wait_stream(main)
                            with torch.cuda.stream(self.commit_stream):
                                for _,_,kk,vv,_,_ in jobs:
                                    kk.record_stream(self.commit_stream)
                                    vv.record_stream(self.commit_stream)
                                self.commit_jobs(jobs)
                            if l==self.last_sparse_layer:
                                main.wait_stream(self.commit_stream)
                        self.pending_clean=[]
                else:
                    self.commit_group(l,model,*self.group_graph(model,k,v),start,layer_enabled)
        cache['global_end_index'].fill_(start+n);cache['local_end_index'].fill_(end)
        return module.o(out.flatten(2))
    def commit_jobs(self,jobs):
        grouped=self.batched_group([(m,kk,vv) for _,m,kk,vv,_,_ in jobs])
        for (layer,m,_,_,pos,encode_layer),values in zip(jobs,grouped):
            self.commit_group(layer,m,*values,pos,encode_layer)

    def commit_group(self,l,model,kg,vg,order,start,layer_enabled):
        store=self.stores[l]
        sg=self.summary_graph(model,kg,vg)
        slots=(torch.arange(72,device=kg.device)+store.write)%self.config.history_pages
        store.commit(kg,vg,sg,order,start)
        if layer_enabled:self.write_encoding(l,self.encode(l,sg),slots=slots)
        if self.stream_attention and not self.legacy_attention:
            from .stream_attention import full_values
            self.value_slabs[l].index_copy_(1,slots,full_values(sg,model.basis,self.haar_basis))
    def finish(self):
        result=super().finish()
        extra=sum(v.numel()*v.element_size() for e in self.encodings.values() for v in e.values())
        slabextra=sum(s.bytes() for s in self.slabs.values())-result['summary_bytes']
        extra+=slabextra+self.workspace.bytes()+(self.B.numel()*self.B.element_size() if self.B is not None else 0)
        extra+=sum(x.numel()*x.element_size() for x in self.projected_basis.values())
        extra+=sum(p.numel()*p.element_size() for net in (self.reader,self.phi) if net is not None for p in net.parameters())
        value_bytes=sum(x.numel()*x.element_size() for x in self.value_slabs.values())
        extra+=self.batch_graph.static_bytes()+value_bytes
        text_cache=getattr(self,'repeated_context',None)
        packed_text_bytes=sum(x.numel()*x.element_size() for cache in text_cache.caches
                              for x in (cache.get('packed_text') or ())) if text_cache else 0
        result['packed_text_bytes']=packed_text_bytes
        extra+=packed_text_bytes
        result['rechecked_mass']=self.batch_graph.partial_mass=='recheck'
        result['resident_route_graphs']=sum(bool(e.get('resident')) for e in self.batch_graph.entries.values())
        if getattr(self,'resident_own',None) is not None:extra+=self.resident_own.numel()*self.resident_own.element_size()
        result['summary_window']=getattr(self,'summary_window',False)
        result['cached_projection']=getattr(self,'cached_projection',False)
        result['fused_exact_tail']=getattr(self.workspace,'fused_exact_tail',False)
        result['strided_tail']=getattr(self.workspace,'strided_tail',False)
        result['direct_risk_bound']=self.batch_graph.direct_risk_bound
        grouping=getattr(self,'batched_group',None)
        result['group_radix']=bool(grouping and getattr(grouping,'groupers',None)
                                   and all(g.fused_sort=='radix' for g in grouping.groupers))
        result.update(implementation='VideoRuntimeV1',matched_probabilities=self.batch_graph.matched_probabilities,fixed_page_sets=self.batch_graph.fixed_page_sets,pset_layers=sorted(self.pset_layers) if self.pset_layers is not None else None,pset_step_ids=sorted(self.pset_step_ids) if self.pset_step_ids is not None else None,pset_heads=self.pset_head_ids.cpu().tolist() if self.pset_head_ids is not None else None,pset_steps=self.pset_steps,route_m16=self.route_m16,coeff_bf16=self.coeff_bf16,cached_logmass=self.cached_logmass,projected_risk=self.projected_risk,legacy_components=sorted(self.legacy_components),static_route=self.static_route,fused_risk=self.fused_risk,split_summary=self.split_summary,stream_attention=self.stream_attention,stream_tail_mode=self.stream_tail_mode,stream_warps=self.stream_warps,bf16_qk=self.bf16_qk,compact_reader=self.batch_graph.compact_reader,tensor_math=self.tensor_math,projection_float=self.batch_graph.projection_float,representative_value_bytes=value_bytes,full_pset=self.full_pset,tile_batch=self.tile_batch,graph_batches=self.graph_batches,compile_route=self.compile_route,compile_tail=self.compile_tail,action_kernel=self.action_kernel,batch_graph_shapes=len(self.batch_graph.entries),batch_graph_replays=self.batch_graph.calls,
            page_encoding='commit-time history; once per step current; shared by every Query tile',
            raw_kv='native resident [1,L,H,D], current scratch writes, no history cat/repack/gather',
            logits=('M16 FP32 R4 for routing; Q64 recomputed inside streamed Attention' if self.stream_attention else 'Q64 FP32 R4 shared with M16 Risk/Reader'),
            attention=('split-K raw page LUT + FP32 full128-mean/rank8 representative Values, online mass/LSE merge' if self.stream_attention else 'repository Page64/Q64 direct-LUT exact + full128 mean/rank8 batched tail + natural-LSE merge'),
            extra_tensor_bytes=extra,workspace_bytes=self.workspace.bytes(),counters=self.counters.cpu().tolist(),validation=self.validation,
            timings=self.timings.finish(),counter_labels=['accepted_swaps','scored_events','nonfinite_scores','encoded_head_pages'],memory_note='Tensor bytes exclude graph pool reservations; CUDA peak counters include graph storage')
        if self.legacy_attention:result['attention']='Previous Page64 direct-LUT exact + full128 mean/rank8 two-term BF16 tail and base2 LSE merge; no representative Value slab'
        result['persistent_bytes']+=extra;result['within_limit']=result['persistent_bytes']<=result['limit_bytes']
        result['status']='engineering_runtime_pending_video_quality'
        return result
