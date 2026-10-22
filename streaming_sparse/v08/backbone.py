"""Differentiable functional replica of the native streaming block forward.

No inference adapter patch, in-place cache writes, mutable cross-attention cache
or checkpoint closure side effects. Every layer returns its new raw KV as values.
"""
import copy
import hashlib
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from .cache import LayerKV,CacheIdentity
from .attention import attend,attend_multi,masks
from .common import protocol,nmse,pair_index,ROOT,sha


def load_teacher(device='cuda'):
    from utils.wan_wrapper import WanDiffusionWrapper
    p=protocol();path=ROOT/p['teacher_checkpoint']
    if sha(path)!=p['teacher_sha256']:raise ValueError('Teacher checkpoint SHA mismatch')
    wrapper=WanDiffusionWrapper(model_name=p['model_name'],is_causal=True,timestep_shift=5.0)
    state=torch.load(path,map_location='cpu',weights_only=False)
    wrapper.load_state_dict(state['generator_ema'],strict=True)
    del state
    wrapper=wrapper.eval().requires_grad_(False).to(device=device,dtype=torch.bfloat16)
    wrapper.model.gradient_checkpointing=False
    return wrapper


def native_steps(wrapper):
    from omegaconf import OmegaConf
    p=protocol();c=OmegaConf.merge(OmegaConf.load(ROOT/p['default_config']),OmegaConf.load(ROOT/p['config']))
    values=torch.tensor(c.denoising_step_list,dtype=torch.long)
    if c.warp_denoising_step:
        lookup=torch.cat((wrapper.scheduler.timesteps.cpu(),torch.zeros(1)))
        values=lookup[1000-values]
    if not torch.allclose(values.float(),torch.tensor(p['native_timesteps']),atol=1e-5,rtol=0):
        raise ValueError('Native scheduler timestep identity changed')
    return values.tolist(),OmegaConf.to_container(c,resolve=True)


def native_cache(wrapper,device,dtype,tokens=32760):
    model=wrapper.model
    return [dict(k=torch.zeros(1,tokens,b.self_attn.num_heads,b.self_attn.head_dim,device=device,dtype=dtype),
                 v=torch.zeros(1,tokens,b.self_attn.num_heads,b.self_attn.head_dim,device=device,dtype=dtype),
                 global_end_index=torch.zeros(1,device=device,dtype=torch.long),
                 local_end_index=torch.zeros(1,device=device,dtype=torch.long)) for b in model.blocks]


class FunctionalBackbone(nn.Module):
    def __init__(self,wrapper,router=None,layers=None,query_microbatch=None):
        super().__init__();self.wrapper=wrapper;self.router=router
        p=protocol();self.layers=tuple(p['layers'] if layers is None else layers)
        self.query_microbatch=query_microbatch or p['query_microbatch']
        self.frames_per_chunk=p['frames_per_chunk']
        self.model_version='initial';self.router_version='initial'
        self._eval_history=None;self._eval_sketch={}
        self._parameter_versions=None;self._revision=0

    @property
    def model(self):return self.wrapper.model

    def identity(self,sample,owner='student'):
        versions=tuple(p._version for p in self.parameters())
        if self._parameter_versions is not None and versions!=self._parameter_versions:
            self._revision+=1;self._eval_history=None;self._eval_sketch.clear()
        self._parameter_versions=versions
        history_id=f"{sample['trajectory_id']}.history{sample['history_tokens']}"
        return CacheIdentity(f'{self.model_version}.revision{self._revision}',self.router_version,sample['trajectory_id'],history_id,
                             0,sample['history_tokens'],sample['context_noise'],owner=owner)

    def _head(self,q,kh,vh,kc,vc,state,aux_state,keep,step,chosen,collect_aux,detach_router,mass_only,collect_mechanisms,layer,head):
        micro=self.query_microbatch;scores=[];delta_norm=[]
        def score_function(q0,ck,cv,st):
            route=self.router.score(q0,ck,cv,st,keep,step,detach_router,mass_only)
            return route['scores'],route['delta'].detach().square().sum()
        for offset in range(0,len(q),micro):
            args=(q[offset:offset+micro],kc,vc,state)
            result=activation_checkpoint(score_function,*args,use_reentrant=False) if torch.is_grad_enabled() else score_function(*args)
            scores.append(result[0]);delta_norm.append(result[1])
        scores=torch.cat(scores)
        # Batched across queries, never pooled: each row solves its OWN identical
        # token constraint. Avoid293 separate64-iteration GPU launch sequences.
        mask,hard,soft=masks(scores,state['costs'],keep)
        actual=(hard*state['costs']).sum(-1)
        diagnostic=torch.stack((actual.sum(),actual.new_tensor(len(q)),
            ((soft<1e-4)|(soft>1-1e-4)).float().sum(),soft.new_tensor(soft.numel()),
            torch.stack(delta_norm).sum(),scores.detach().square().sum()))
        # Native BF16 executes the whole head, preserving its query-length
        # backend arithmetic. Router representations still use16-query batches.
        la=q.new_zeros((),dtype=torch.float32);lc=la.clone();records=[];aux_count=0
        def execution(q0,hk,hv,ck,cv,m):return attend(q0,hk,hv,ck,cv,m,self.router.sketch.page_size).to(cv.dtype)
        def auxiliary(q0,hk,hv,ck,cv,st,m):
            e=self.router.sketch.estimate(q0.detach(),st,ck.detach(),cv.detach() if self.router.kind=='V' else None,self.router.kind)
            if not detach_router:
                m=self.router(q0.detach(),ck.detach(),cv.detach(),st,keep,step,True,mass_only)['mask']
            with torch.no_grad():
                dense=attend(q0.detach(),hk.detach(),hv.detach(),ck.detach(),cv.detach(),torch.ones_like(m),self.router.sketch.page_size)
            pred=attend(q0.detach(),hk.detach(),hv.detach(),ck.detach(),cv.detach(),m,self.router.sketch.page_size)
            return nmse(pred,dense),self.router.sketch.calibration_loss(q0.detach(),hk.detach(),ck.detach(),st,e)
        for index,offset in enumerate(range(0,len(q),micro)):
            qq=q[offset:offset+micro];mm=mask[offset:offset+micro]
            if collect_aux and index in chosen:
                args=(qq,kh,vh,kc,vc,aux_state,mm)
                aa,cc=activation_checkpoint(auxiliary,*args,use_reentrant=False) if torch.is_grad_enabled() else auxiliary(*args)
                la=la+aa;lc=lc+cc;aux_count+=1
            if collect_mechanisms and index in chosen:
                from .diagnostics import mechanisms
                route=self.router.score(qq,kc,vc,state,keep,step,detach_router,mass_only)
                route.update(hard=hard[offset:offset+micro],soft=soft[offset:offset+micro],mask=mm)
                record=mechanisms(self.router,qq.detach(),kh.detach(),vh.detach(),kc.detach(),vc.detach(),state,route,keep)
                records.append(dict(layer=layer,head=head,query_offset=offset,query_count=len(qq),**record))
        return mask,la,lc,diagnostic,aux_count,records

    def _block(self,block,layer,x,e0,grid,freqs,context,history_k,history_v,start,sparse,step,keep,
               collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity):
        from wan.modules.causal_model import causal_rope_apply
        frames=e0.shape[1];tokens=x.shape[1]//frames
        e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2)
        mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2)
        att=block.self_attn;b,s=mod.shape[:2];h=att.num_heads;d=att.head_dim
        q=att.norm_q(att.q(mod)).view(b,s,h,d)
        k=att.norm_k(att.k(mod)).view(b,s,h,d);v=att.v(mod).view(b,s,h,d)
        q=causal_rope_apply(q,grid,freqs,start_frame=start).type_as(v)
        k=causal_rope_apply(k,grid,freqs,start_frame=start).type_as(v)
        aux_att=x.new_zeros((),dtype=torch.float32);aux_cal=aux_att.clone()
        # Counters are returned, never appended to shared checkpoint state.
        diagnostic=x.new_zeros(6,dtype=torch.float32);aux_count=0;records=[]
        if not sparse or layer not in self.layers or not history_k.shape[1] or self.router is None:
            all_k=torch.cat((history_k,k),1);all_v=torch.cat((history_v,v),1)
            y=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),all_k.transpose(1,2),
                all_v.transpose(1,2),dropout_p=0.,is_causal=False).transpose(1,2)
        else:
            head_outputs=[]
            for head in range(h):
                pair=self.layers.index(layer)*h+head
                kh=history_k[0,:,head];vh=history_v[0,:,head]
                key=(cache_identity,layer,head)
                if not torch.is_grad_enabled() and key in self._eval_sketch:
                    state=self._eval_sketch[key]
                else:
                    state=self.router.build(kh,vh,pair,detach_inputs=detach_router)
                    if not torch.is_grad_enabled():self._eval_sketch[key]=state
                aux_state=self.router.build(kh,vh,pair,detach_inputs=True) if collect_aux and not detach_router else state
                micro=self.query_microbatch;batches=math.ceil(s/micro)
                rng=torch.Generator().manual_seed(aux_seed+pair)
                full_batches=max(1,s//micro)
                chosen=set(torch.randperm(full_batches,generator=rng)[:min(8,full_batches)].tolist())
                yy,la,lc,dd,count,head_records=self._head(q[0,:,head],kh,vh,k[0,:,head],v[0,:,head],state,aux_state,
                    keep,step,chosen,collect_aux,detach_router,mass_only,collect_mechanisms,layer,head)
                head_outputs.append(yy);aux_att=aux_att+la;aux_cal=aux_cal+lc;diagnostic=diagnostic+dd;aux_count+=count
                records.extend(head_records)
            y=attend_multi(q[0],history_k[0],history_v[0],k[0],v[0],torch.stack(head_outputs),self.router.sketch.page_size)[None]
        x=x+(att.o(y.flatten(2)).unflatten(1,(frames,tokens))*e[2]).flatten(1,2)
        # No cross-attention mutable cache: recomputation is pure.
        x=x+block.cross_attn(block.norm3(x),context,None,crossattn_cache=None)
        ff=block.ffn((block.norm2(x).unflatten(1,(frames,tokens))*(1+e[4])+e[3]).flatten(1,2))
        x=x+(ff.unflatten(1,(frames,tokens))*e[5]).flatten(1,2)
        return x,k,v,aux_att,aux_cal,diagnostic,x.new_tensor(float(aux_count),dtype=torch.float32),records

    def forward_chunk(self,latent,embeds,timestep,cache,start,sparse=True,step=0,keep=.2,identity=None,
                      checkpoint=True,collect_aux=True,detach_router=True,mass_only=False,collect_mechanisms=False):
        from wan.modules.model import sinusoidal_embedding_1d
        model=self.model
        if collect_mechanisms and torch.is_grad_enabled():raise ValueError('Mechanism probes are evaluation-only')
        if latent.shape[0]!=1 or model.model_type!='t2v':raise ValueError('v08 supports T2V microbatch1')
        device=next(model.parameters()).device;latent=latent.to(device)
        x=model.patch_embedding(latent.permute(0,2,1,3,4))
        grid=torch.tensor([x.shape[2:]],dtype=torch.long)
        x=x.flatten(2).transpose(1,2)
        t=torch.full(latent.shape[:2],float(timestep),device=device,dtype=torch.float32)
        e=model.time_embedding(sinusoidal_embedding_1d(model.freq_dim,t.flatten()).type_as(x))
        e0=model.time_projection(e).unflatten(1,(6,model.dim)).unflatten(0,t.shape)
        if embeds.shape[0]!=1 or embeds.shape[1]>model.text_len:raise ValueError('Invalid prompt embedding')
        context=model.text_embedding(torch.nn.functional.pad(embeds.to(device),(0,0,0,model.text_len-embeds.shape[1])))
        freqs=model.freqs.to(device)
        if identity is None:
            identity=CacheIdentity(self.model_version,self.router_version,'synthetic','synthetic',0,
                                   start*int(grid[0,1]*grid[0,2]),0.)
        if cache:
            from .cache import validate
            # During context construction cache is a prefix of final history.
            for entry in cache:
                if entry.identity!=identity or entry.k.shape[1]!=start*int(grid[0,1]*grid[0,2]):
                    raise ValueError('Cache identity/history prefix mismatch')
        aux_seed=int.from_bytes(hashlib.sha256(identity.sample_id.encode()).digest()[:4],'little')
        new_cache=[];atts=[];cals=[];counts=[];diagnostics=[];mechanism_records=[]
        for layer,block in enumerate(model.blocks):
            if cache:kh,vh=cache[layer].k,cache[layer].v
            else:
                kh=x.new_empty((1,0,block.self_attn.num_heads,block.self_attn.head_dim));vh=kh
            # Bind all loop values NOW: backward recomputation must not see the last layer.
            def function(xx,ee,cc,hk,hv,block=block,layer=layer):
                return self._block(block,layer,xx,ee,grid,freqs,cc,hk,hv,start,sparse,step,keep,
                    collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,identity)
            if checkpoint and torch.is_grad_enabled():
                values=activation_checkpoint(function,x,e0,context,kh,vh,use_reentrant=False)
            else:values=function(x,e0,context,kh,vh)
            x,k,v,la,lc,diag,count,records=values
            new_cache.append(LayerKV(torch.cat((kh,k),1),torch.cat((vh,v),1),identity,layer,k,v))
            atts.append(la);cals.append(lc);counts.append(count);diagnostics.append(diag)
            mechanism_records.extend(records)
        flow=torch.stack(model.unpatchify(model.head(x,e.unflatten(0,t.shape).unsqueeze(2)),grid)).permute(0,2,1,3,4)
        x0=self.wrapper._convert_flow_pred_to_x0(flow.flatten(0,1),latent.flatten(0,1),t.flatten()).unflatten(0,flow.shape[:2])
        count=torch.stack(counts).sum().clamp_min(1)
        return dict(x0=x0,flow=flow,cache=tuple(new_cache),att_aux=torch.stack(atts).sum()/count,
                    cal_aux=torch.stack(cals).sum()/count,diagnostics=torch.stack(diagnostics).sum(0),mechanisms=mechanism_records)

    def replay(self,sample,keep=.2,sparse=True,recent_gradient=True,checkpoint=True,collect_aux=True,
               detach_router=True,audit=False,mass_only=False,collect_mechanisms=False):
        from .cache import build_history
        ident=self.identity(sample,owner='teacher' if self.router is None else 'student')
        device=next(self.model.parameters()).device;dtype=next(self.model.parameters()).dtype
        h=sample['history'].to(device=device,dtype=dtype);x=sample['noisy'].to(device=device,dtype=dtype)
        embeds=sample['prompt_embeds'].to(device=device,dtype=dtype)
        token=(sample['history'].untyped_storage().data_ptr(),sample['history']._version,
               sample['prompt_embeds'].untyped_storage().data_ptr(),sample['prompt_embeds']._version,
               sample.get('payload_sha256'))
        if self._eval_history is not None and (self._eval_history[0]!=ident or self._eval_history[2]!=token):
            self._eval_history=None;self._eval_sketch.clear()
        if not torch.is_grad_enabled() and self._eval_history is not None:
            cache=self._eval_history[1]
        else:
            cache=build_history(self,h,embeds,ident,recent_gradient,checkpoint,audit)
            if not torch.is_grad_enabled():self._eval_history=(ident,cache,token)
            else:self._eval_history=None;self._eval_sketch.clear()
        result=self.forward_chunk(x,embeds,sample['timestep'],cache,h.shape[1],sparse=sparse,
            step=sample['step'],keep=keep,identity=ident,checkpoint=checkpoint,collect_aux=collect_aux,detach_router=detach_router,mass_only=mass_only,collect_mechanisms=collect_mechanisms)
        if audit:result['history_cache']=cache
        return result
