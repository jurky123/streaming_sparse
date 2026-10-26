"""Temporal-mask reuse experiments and a NONDEPLOYABLE same-trajectory oracle.

The oracle is only a timing bound. Real temporal policies use earlier steps
of the current generation; their cache is cleared by every set_mode call.
"""
import os
import torch
from ..fast_network import FastNetwork
from .. import fast_network
from ..opt6_guarded import load_guarded_head
class ReuseNetwork(FastNetwork):
    def set_mode(self,kind):
        super().set_mode(kind)
        if getattr(self,'reuse_policy','')!='oracle':self.route_cache={}

    def _block(self,block,layer,x,e0,grid,freqs,context,history_k,history_v,start,sparse,step,keep,
               collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity):
        if not sparse or layer not in self.layers or not history_k.shape[1]or self.router.kind=='Dense':
            return super()._block(block,layer,x,e0,grid,freqs,context,history_k,history_v,start,False,step,keep,
                collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity)
        if torch.is_grad_enabled()or collect_aux:raise RuntimeError('Evaluation only')
        if torch.backends.cuda.matmul.allow_tf32:raise RuntimeError('Router requires strict FP32')
        from wan.modules.causal_model import causal_rope_apply
        frames=e0.shape[1];tokens=x.shape[1]//frames
        e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2)
        mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2)
        att=block.self_attn;b,s=mod.shape[:2];h=att.num_heads;d=att.head_dim
        q=att.norm_q(att.q(mod)).view(b,s,h,d);k=att.norm_k(att.k(mod)).view(b,s,h,d)
        v=att.v(mod).view(b,s,h,d)
        q=causal_rope_apply(q,grid,freqs,start_frame=start).type_as(v)
        k=causal_rope_apply(k,grid,freqs,start_frame=start).type_as(v)
        kh=history_k[0].transpose(0,1);vh=history_v[0].transpose(0,1)
        kc=k[0].transpose(0,1);vc=v[0].transpose(0,1);qq=q[0].transpose(0,1)
        raw_k=torch.cat((kh,kc),1);raw_v=torch.cat((vh,vc),1)
        key=(self.router.kind,layer,start,step if self.reuse_policy=='oracle' else 0)
        cached_route=self.route_cache.get(key)
        reuse=cached_route is not None and (self.reuse_policy=='oracle' or step not in self.refresh_steps)
        if reuse:
            mask,sketch=cached_route;prepared=None
        else:
            history_key=(layer,self.router.kind)
            identity=(cache_identity,history_k.data_ptr())
            cached=self._eval_sketch.get(history_key)if self.history_cache else None
            if cached is not None and cached[0]==identity:
                from ..core import part,Sketch
                previous=cached[1];parameters=self.router.states[str(layer)].dictionary()
                current=self.make_part(kc,vc if self.router.kind=='V32-Set'else None,parameters['current_weight'],parameters['current_bias'])
                sketch=Sketch(previous.parts[:-1]+[current],previous.costs,previous.origins,previous.history_tokens,
                    previous.current_tokens,previous.key_basis,previous.value_basis)
            else:
                sketch=self.make_build(kh,vh,kc,vc,self.router.states[str(layer)].dictionary(),need_value=self.router.kind=='V32-Set')
                if self.history_cache:self._eval_sketch[history_key]=(identity,sketch)
            prepared=self.prepare(sketch,self.router.kind)
            routed_q=qq[:,::self.query_stride]
            if self.graph_route is not None:
                mask=self.graph_route(layer,routed_q,sketch,prepared,step,self.router)
            else:
                masks=[]
                for offset in range(0,routed_q.shape[1],self.route_batch):
                    qt=routed_q[:,offset:offset+self.route_batch]
                    masks.append(self.fast_route(layer,qt,sketch,prepared,step,self.router))
                mask=torch.cat(masks,1)
            if self.query_stride!=1:mask=mask.repeat_interleave(self.query_stride,dim=1)[:,:s].contiguous()
            from ..core import Sketch
            frozen=Sketch([],sketch.costs,sketch.origins,sketch.history_tokens,sketch.current_tokens,sketch.key_basis,sketch.value_basis)
            self.route_cache[key]=(mask.clone(),frozen)
        if self.attention_backend=='triton':
            y=fast_network.sparse_attention(qq,raw_k,raw_v,mask,self.attention_group)
        elif self.attention_backend=='flex':
            from ..fast_flex import flex_attention
            y=flex_attention(qq,raw_k,raw_v,mask)
        else:
            # Exact original SDPA tiling, for isolating routing-only numerical effects.
            y=torch.cat([torch.nn.functional.scaled_dot_product_attention(qq[:,o:o+128][None],
                raw_k[None],raw_v[None],attn_mask=mask[:,o:o+128].repeat_interleave(sketch.costs.long(),-1)[None])[0]
                for o in range(0,s,128)],1)
        if self.inspect_callback is not None:
            self.inspect_callback(layer,start,step,qq,raw_k,raw_v,sketch,prepared,mask,y)
        y=y.transpose(0,1)[None]
        x=x+(att.o(y.flatten(2)).unflatten(1,(frames,tokens))*e[2]).flatten(1,2)
        x=x+block.cross_attn(block.norm3(x),context,None,crossattn_cache=None)
        ff=block.ffn((block.norm2(x).unflatten(1,(frames,tokens))*(1+e[4])+e[3]).flatten(1,2))
        x=x+(ff.unflatten(1,(frames,tokens))*e[5]).flatten(1,2)
        # Leave diagnostics on device. Transfer once after the trajectory, never
        # use GPU tensor booleans or per-tile .item() in the hot path.
        nh=sketch.costs.shape[0]-74
        count=mask*sketch.costs[None,None];rh=count[...,:nh].sum();rc=count[...,nh:].sum();nq=h*s
        self.routing_records.append(dict(layer=layer,start=start,step=step,queries=nq,
            history_tokens=kh.shape[1],current_tokens=kc.shape[1],retained_H=rh,retained_C=rc,
            keep=(rh+rc)/(nq*(kh.shape[1]+kc.shape[1])),keep_H=rh/(nq*kh.shape[1]),keep_C=rc/(nq*kc.shape[1])))
        zero=x.new_zeros((),dtype=torch.float32)
        diagnostic=torch.stack((rh+rc,rh.new_tensor(nq),rh,rc,rh.new_tensor(nq*(kh.shape[1]+kc.shape[1])),rh.new_tensor(1)))
        return x,k,v,zero,zero,diagnostic,zero,[]


def load_reuse(directory,training_seed=0,**options):
    net,m=load_guarded_head(directory,training_seed,**options)
    net.__class__=ReuseNetwork
    net.reuse_policy=os.environ.get('SLIMAV_REUSE','02')
    net.refresh_steps=tuple(int(v)for v in net.reuse_policy)if net.reuse_policy!='oracle' else ()
    if net.reuse_policy!='oracle' and 0 not in net.refresh_steps:raise ValueError('Step 0 must refresh')
    net.route_cache={}
    return net,m

def main():
    from .. import fast_bench
    fast_bench.load_fast=load_reuse;fast_bench.main()
if __name__=='__main__':main()
