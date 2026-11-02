"""Actual closed-loop sparse attention: physical KV ring and clean-only summaries."""
import math,types
from pathlib import Path
import torch
from .broad_common import Engine
from .response_hard import features,balanced
from .cache_grouping import reorder
from .cache_sparse_prototype import load_kernel

METHODS=['dense','dense_trim','dense_grouped','mean','mass_mean','mass_var']
ROOT=Path('artifacts/rollout_acceptance_20261030')

class Runtime:
    def __init__(self,engine,method,profile=False,audit=False,fused=True):
        self.engine=engine;self.fused=fused;self.method=method;self.profile=profile;self.audit=audit
        self.grouped=method in ['dense_grouped','mean','mass_mean','mass_var']
        self.sparse=method in ['mean','mass_mean','mass_var']
        self.cap=360 if method=='dense' else 338
        self.original=[b.self_attn.forward for b in engine.wrapper.model.blocks]
        self.bank={};self.weights={};self.alpha={};self.pk={};self.events=[];self.checks=[]
        self.n=[0]*30;self.write=[0]*30;self.commits=[0]*30;self.sparse_calls=0
        if self.grouped:
            bank=torch.load('artifacts/broad_coverage_20261030/dictionary.pt',weights_only=True)
            saved=torch.load('artifacts/mass_variance_20261030/train_00_30/best.pt',weights_only=False,map_location='cpu')['models']
            for l in range(1,30):
                self.bank[l]=bank[l].cuda()
                self.pk[l]=torch.empty(12,self.cap,4,128,device='cuda',dtype=torch.float32)
                if method.startswith('mass'):
                    self.weights[l]=saved[method][l]['weight'].cuda()
                    if method=='mass_var':self.alpha[l]=saved[method][l]['alpha'].cuda()
        # Evicted prefix of a partial oldest chunk is spread over response pages.
        early=torch.div(torch.arange(22)*72,22,rounding_mode='floor').tolist()
        self.page_order=torch.tensor(early+[i for i in range(72) if i not in early],device='cuda') if self.grouped else torch.empty(0,device='cuda')
        self.offsets=torch.arange(64,device='cuda') if self.grouped else torch.empty(0,device='cuda')
        self.kernel=load_kernel() if self.sparse else None
        for l,b in enumerate(engine.wrapper.model.blocks):
            tokens=(360 if l==0 else self.cap)*64+4608
            c=engine.cache[l]
            if c['k'].shape[1]!=tokens:
                c['k']=torch.empty(1,tokens,12,128,device='cuda',dtype=torch.bfloat16)
                c['v']=torch.empty_like(c['k'])
            b.self_attn.max_attention_size=tokens
            if self.grouped and l>0:
                def forward(module,x,seq_lens,grid_sizes,freqs,block_mask,kv_cache=None,current_start=0,cache_start=None,index=l):
                    return self.forward(index,module,x,grid_sizes,freqs,kv_cache,current_start)
                b.self_attn.forward=types.MethodType(forward,b.self_attn)
        engine.reset()
        self.persistent_bytes=sum(c[k].numel()*c[k].element_size() for c in engine.cache for k in ['k','v'])
        self.persistent_bytes+=sum(x.numel()*x.element_size() for bank in [self.bank,self.weights,self.alpha,self.pk] for x in bank.values())
        self.persistent_bytes+=(self.page_order.numel()*8+self.offsets.numel()*8) if self.grouped else 0
        self.budget=30*27648*12*128*2*2
        assert self.persistent_bytes<=self.budget,(self.method,self.persistent_bytes,self.budget)

    def timed(self,name,fn):
        if not self.profile:return fn()
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();out=fn();b.record();self.events.append((name,a,b));return out

    def reset(self):
        self.engine.reset();self.n=[0]*30;self.write=[0]*30;self.commits=[0]*30
        self.events=[];self.checks=[];self.sparse_calls=0

    def aggregate(self,k,l):
        groups=k.reshape(12,72,4,16,128)
        if l not in self.weights:return groups.mean(3)
        x=groups-groups.mean(3,keepdim=True)
        a=(2*torch.tanh(torch.einsum('hbrtd,hrd->hbrt',x,self.weights[l])/2)).softmax(-1)
        return torch.einsum('hbrt,hbrtd->hbrd',a,groups)

    def commit(self,l,k,v,cache):
        k=k[0].transpose(0,1).float();v=v[0].transpose(0,1)
        def group():
            x=features(k,v,self.bank[l],'response');order=balanced(x,[3,3,2,2,2])
            xp=reorder(x,order);inner=balanced(xp.reshape(-1,64,32),[2,2]).reshape(12,72,64)
            order=order.reshape(12,72,64).gather(2,inner)[:,self.page_order].flatten(1)
            return reorder(k,order),reorder(v,order)
        kr,vr=self.timed('group_reorder',group)
        from .rollout_kernels import aggregate
        proto=self.timed('aggregate',lambda:aggregate(kr,self.weights.get(l)) if self.fused else self.aggregate(kr,l))
        if self.audit:
            torch.testing.assert_close(proto,self.aggregate(kr,l),rtol=3e-4,atol=3e-5)
        def store():
            slots=(torch.arange(72,device='cuda')+self.write[l])%self.cap
            tokens=(slots[:,None]*64+self.offsets).flatten()
            cache['k'][0].index_copy_(0,tokens,kr.transpose(0,1).to(torch.bfloat16))
            cache['v'][0].index_copy_(0,tokens,vr.transpose(0,1))
            self.pk[l].index_copy_(1,slots,proto)
            self.write[l]=(self.write[l]+72)%self.cap;self.n[l]=min(self.cap,self.n[l]+72);self.commits[l]+=1
        self.timed('ring_commit',store)

    def route(self,l,q,p):
        def reference():
            full=q[0].transpose(0,1).float().reshape(12,72,64,128)
            mean=full.mean(2);flat=self.pk[l][:,:p].flatten(1,2)
            logits=(mean@flat.transpose(1,2)/math.sqrt(128)).reshape(12,72,p,4)
            if l in self.alpha:
                variance=full.var(2,unbiased=False)
                corr=(variance@flat.square().transpose(1,2)/256).reshape_as(logits)
                logits+=self.alpha[l][:,None,None,None]*corr
            return logits.logsumexp(-1)
        if self.fused:
            from .rollout_kernels import route_scores
            z=route_scores(q,self.pk[l][:,:p],self.alpha.get(l))
            if self.audit:
                torch.testing.assert_close(z,reference(),atol=5e-4,rtol=5e-4)
        else:z=reference()
        selected=z.topk(72,-1).indices
        current=torch.arange(p,p+72,device='cuda').expand(12,72,72)
        return torch.cat((selected,current),-1).sort(-1).values[None].contiguous()

    @torch.no_grad()
    def forward(self,l,module,x,grid_sizes,freqs,cache,start):
        from wan.modules.causal_model import causal_rope_apply
        b,s,_=x.shape;assert b==1 and s==4608
        q=module.norm_q(module.q(x)).view(b,s,12,128)
        k=module.norm_k(module.k(x)).view(b,s,12,128);v=module.v(x).view(b,s,12,128)
        q=causal_rope_apply(q,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        k=causal_rope_apply(k,grid_sizes,freqs,start_frame=start//1536).to(v.dtype).contiguous()
        p=self.n[l];step=self.engine.current['step'];chunk=self.engine.current['chunk']
        def write_current():
            cache['k'][:,p*64:p*64+s].copy_(k);cache['v'][:,p*64:p*64+s].copy_(v)
        self.timed('current_write',write_current)
        kh=cache['k'][:,:p*64+s];vh=cache['v'][:,:p*64+s]
        if step==4 or p<=72 or not self.sparse:
            out=self.timed('clean_attention' if step==4 else 'dense_attention',lambda:self.engine.dense(q,kh,vh))
        else:
            lut=self.timed('route',lambda:self.route(l,q,p))
            def attend():
                out=torch.empty_like(q);lse=torch.empty(1,12,s,device='cuda',dtype=torch.float32)
                self.kernel[(72,12)](q,kh,vh,128**-.5,144,lut,lse,out,12,s,p*64+s,72,128,64,64,num_warps=4,num_stages=3)
                return out
            out=self.timed('sparse_attention',attend);self.sparse_calls+=1
            if self.audit and step==0 and chunk in [2,5,6] and l in [1,14,29]:
                errors=[]
                for head in [0,4,8,11]:
                    for qi in [0,1024,3072,4607]:
                        ids=(lut[0,head,qi//64,:,None]*64+self.offsets).flatten()
                        kk=kh[0,:,head].index_select(0,ids).float();vv=vh[0,:,head].index_select(0,ids).float()
                        target=(q[0,qi,head].float()@kk.T/math.sqrt(128)).softmax(-1)@vv
                        error=(out[0,qi,head].float()-target).square().sum()/target.square().sum().clamp_min(1e-8)
                        assert error<1e-4,error
                        errors.append(error.item())
                self.checks.append(dict(layer=l,chunk=chunk,max_relative_mse=max(errors)))
        if step==4:
            # Only now is this chunk immutable. Never summarize noisy current KV.
            self.commit(l,k,v,cache)
        cache['global_end_index'].fill_(start+s);cache['local_end_index'].fill_(p*64+s)
        return module.o(out.flatten(2))

    def finish(self):
        torch.cuda.synchronize();times={}
        for name,a,b in self.events:times[name]=times.get(name,0.)+a.elapsed_time(b)
        return dict(method=self.method,history_pages=self.cap,selected_history_pages=72,
            persistent_self_kv_and_aux_bytes=self.persistent_bytes,limit_bytes=self.budget,
            commits=self.commits,sparse_calls=self.sparse_calls,profile_gpu_ms=times,checks=self.checks)

    def close(self):
        for b,forward in zip(self.engine.wrapper.model.blocks,self.original):b.self_attn.forward=forward
