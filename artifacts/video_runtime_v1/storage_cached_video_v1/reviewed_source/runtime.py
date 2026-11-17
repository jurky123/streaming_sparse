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
    def __init__(self,engine,model,reader,phi,B,*,full_pset=False,tile_batch=2,profile=False,config=Config(),validate=False,graph_batches=True,compile_route=False,compile_tail=False,action_kernel=False,tensor_math=False,projection_float=False,compact_reader=False,stream_attention=False,stream_tail_mode=4,stream_warps=4,bf16_qk=False,split_summary=False,fused_risk=False,static_route=False,legacy_components=None,projected_risk=False,cached_logmass=False,coeff_bf16=False):
        if tile_batch<1 or 72%tile_batch:raise ValueError('tile_batch must divide 72')
        if config.probes!=16 or config.density!=.1:raise ValueError('Frozen configuration requires M16 / 10%')
        # The deployed default uses the prior direct-LUT exact and full-mean tail.
        # An explicit string (including "") preserves historical controls.
        if legacy_components is None:legacy_components="attention"
        self.legacy_components=set(legacy_components.split(",")) if legacy_components else set()
        self.legacy_attention="attention" in self.legacy_components
        self.stream_tail_mode=stream_tail_mode;self.stream_warps=stream_warps;self.bf16_qk=bf16_qk;self.stream_attention=stream_attention;self.value_slabs={};self.reader=reader;self.phi=phi;self.B=B.double();self.full_pset=full_pset;self.tile_batch=tile_batch
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
        self.batch_graph=BatchGraphs(risk_fn,phi,config,compile_route=compile_route,compile_tail=compile_tail,action_kernel=action_kernel,tensor_math=tensor_math,projection_float=projection_float,compact_reader=compact_reader,probes_only=stream_attention,bf16_qk=bf16_qk,static_route=static_route)
        self.tensor_math=tensor_math;self.compile_route=compile_route;self.compile_tail=compile_tail;self.action_kernel=action_kernel
        self.validate=validate;self.validation=[];self.checked=set();self.slabs={};self.encodings={};self.projected_basis={}
        super().__init__(engine,model,config)
        h=12;p=config.history_pages+72;device=self.B.device
        self.haar_basis=haar(self.B);self.offsets=torch.linspace(0,63,16,device=device).round().long()
        for l,store in self.stores.items():
            old=store.summary
            slab=Summary(*(torch.empty((h,p,*getattr(old,n).shape[2:]),device=device,dtype=torch.bfloat16) for n in ('keys','mean','detail','stats')))
            self.slabs[l]=slab;store.summary=slab.slice(0,config.history_pages)
            self.encodings[l]=dict(co=torch.empty(h,p,5,17,device=device,dtype=torch.bfloat16 if coeff_bf16 else torch.float32),A=torch.empty(h,p,8,4,device=device,dtype=torch.bfloat16 if coeff_bf16 else torch.float32),mean=torch.empty(h,p,16,device=device,dtype=torch.float64),detail=torch.empty(h,p,3,16,device=device,dtype=torch.float64))
            self.projected_basis[l]=self.model[l].basis.bfloat16().double().transpose(1,2)@self.B
            if stream_attention and not self.legacy_attention:self.value_slabs[l]=torch.empty(12,p,4,128,device=device,dtype=torch.float32)
        self.counters=torch.zeros(4,device=device,dtype=torch.int64)
    def reset(self):
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
        q=module.norm_q(module.q(x)).view(b,n,12,128);k=module.norm_k(module.k(x)).view_as(q);v=module.v(x).view_as(q)
        if "rope" in self.legacy_components:
            from ..review_v1.fused_rope import apply
            q,k=apply(q,k,grid_sizes,freqs,start//1536)
        else:
            q=causal_rope_apply(q,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
            k=causal_rope_apply(k,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        end=p*64+n;cache['k'][:,p*64:end].copy_(k);cache['v'][:,p*64:end].copy_(v)
        kh=cache['k'][:,:end];vh=cache['v'][:,:end];step=self.engine.current['step']
        if not self.sparse or step==4 or p<=72:
            with self.timings.stage('attention'):out=self.engine.dense(q,kh,vh)
        else:
            with self.timings.stage('summary_and_encoding'):
                if getattr(self,"inline_summary",False) or torch.cuda.is_current_stream_capturing():
                    from ..review_v1.ops import SummaryOps
                    current=SummaryOps.build(model,k[0].transpose(0,1).reshape(12,72,64,128),v[0].transpose(0,1).reshape(12,72,64,128),True,haar_basis=self.haar_basis)
                else:
                    current=self.summary_graph(model,k[0].transpose(0,1).reshape(12,72,64,128),v[0].transpose(0,1).reshape(12,72,64,128),True)
                slab=self.slabs[l]
                for name in ('keys','mean','detail','stats'):getattr(slab,name)[:,p:p+72].copy_(getattr(current,name))
                if self.full_pset:self.write_encoding(l,self.encode(l,current),start=p)
                if self.stream_attention and not self.legacy_attention:
                    from .stream_attention import full_values
                    self.value_slabs[l][:,p:p+72].copy_(full_values(current,model.basis,self.haar_basis))
            s=self.slabs[l].slice(0,p+72);en={name:value[:,:p+72] for name,value in self.encodings[l].items()}
            result=self.workspace.tensor('result',q.shape,q.device,q.dtype)
            for first in range(0,72,self.tile_batch):
                t=self.tile_batch;qq=q[:,first*64:(first+t)*64];query=qq[0].transpose(0,1).float()
                with self.timings.stage('route_and_correction'):
                    own=(torch.arange(first,first+t,device=q.device)+p)[None].expand(12,-1)
                    if self.graph_batches:
                        entry=self.batch_graph.get(query.reshape(12,t,64,128),s,en,model.theta,model.basis,own,self.full_pset)
                        if first==0:self.batch_graph.stage(entry,s,en,model.theta,model.basis)
                        ids,mask,risk,stats=self.batch_graph.replay(entry,query.reshape(12,t,64,128),own)
                        self.counters[:3]+=stats;count=ids.shape[-1]
                    else:
                        logits=torch.bmm(query,s.keys.float().flatten(1,2).transpose(1,2)).reshape(12,t,64,p+72,4)/math.sqrt(128)
                        probes=query.reshape(12,t,64,128).index_select(2,self.offsets)
                        z=logits.index_select(2,self.offsets);risk=risk_from_logits(probes,z,s,model.theta)
                        own=(torch.arange(first,first+t,device=q.device)+p)[None].expand(12,-1)
                        sel=RouteOps.select(risk,own,self.config,check_valid=False);mask=sel['mask'];count=max(1,math.floor((p+72)*.1))
                        if self.full_pset:
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
                    result[:,first*64:(first+t)*64].copy_(mixed)
                key=(l,self.engine.current['chunk'],step)
                if self.validate and first==0 and key not in self.checked:
                    from ..review_v1.runtime import SelectedAttention
                    from ..review_v1.ops import RouteOps as ReferenceRoute
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
                kg,vg,order=self.group_graph(model,k,v);sg=self.summary_graph(model,kg,vg)
                slots=(torch.arange(72,device=q.device)+store.write)%self.config.history_pages
                store.commit(kg,vg,sg,order,start)
                if self.full_pset:self.write_encoding(l,self.encode(l,sg),slots=slots)
                if self.stream_attention and not self.legacy_attention:
                    from .stream_attention import full_values
                    self.value_slabs[l].index_copy_(1,slots,full_values(sg,model.basis,self.haar_basis))
        cache['global_end_index'].fill_(start+n);cache['local_end_index'].fill_(end)
        return module.o(out.flatten(2))
    def finish(self):
        result=super().finish()
        extra=sum(v.numel()*v.element_size() for e in self.encodings.values() for v in e.values())
        slabextra=sum(s.bytes() for s in self.slabs.values())-result['summary_bytes']
        extra+=slabextra+self.workspace.bytes()+self.B.numel()*self.B.element_size()
        extra+=sum(x.numel()*x.element_size() for x in self.projected_basis.values())
        extra+=sum(p.numel()*p.element_size() for net in (self.reader,self.phi) for p in net.parameters())
        value_bytes=sum(x.numel()*x.element_size() for x in self.value_slabs.values())
        extra+=self.batch_graph.static_bytes()+value_bytes
        result.update(implementation='VideoRuntimeV1',coeff_bf16=self.coeff_bf16,cached_logmass=self.cached_logmass,projected_risk=self.projected_risk,legacy_components=sorted(self.legacy_components),static_route=self.static_route,fused_risk=self.fused_risk,split_summary=self.split_summary,stream_attention=self.stream_attention,stream_tail_mode=self.stream_tail_mode,stream_warps=self.stream_warps,bf16_qk=self.bf16_qk,compact_reader=self.batch_graph.compact_reader,tensor_math=self.tensor_math,projection_float=self.batch_graph.projection_float,representative_value_bytes=value_bytes,full_pset=self.full_pset,tile_batch=self.tile_batch,graph_batches=self.graph_batches,compile_route=self.compile_route,compile_tail=self.compile_tail,action_kernel=self.action_kernel,batch_graph_shapes=len(self.batch_graph.entries),batch_graph_replays=self.batch_graph.calls,
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
