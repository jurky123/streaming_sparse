"""Optimized evaluation adapter, using the original trained router and backbone."""
import torch
from .network import Network,OnlineRouter
from .common import config
from .fast_route import FastRoute,prepare
from .fast_kernels import sparse_attention
from streaming_sparse.v08.backbone import load_teacher


class FastNetwork(Network):
    def __init__(self,wrapper,router,layers,compile=True,batch=512,attention='triton',group=16,graph=False,
                 fused=False,query_stride=1,history_cache=False,compiled_build=False):
        super().__init__(wrapper,router,layers)
        self.fast_route=FastRoute(compile,fused,query_stride,batch);self.route_batch=batch;self.attention_backend=attention;self.attention_group=group
        self.query_stride=query_stride;self.history_cache=history_cache
        from .core import part,build
        self.make_part=torch.compile(part,fullgraph=True,dynamic=False)if compiled_build else part
        self.make_build=torch.compile(build,fullgraph=True,dynamic=False)if compiled_build else build
        self.prepare=torch.compile(prepare,fullgraph=True,dynamic=False)if compiled_build else prepare
        self.inspect_callback=None
        from .fast_graph import GraphCache
        self.graph_route=GraphCache(self.fast_route,batch)if graph else None

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
        history_key=(layer,self.router.kind)
        identity=(cache_identity,history_k.data_ptr())
        cached=self._eval_sketch.get(history_key)if self.history_cache else None
        if cached is not None and cached[0]==identity:
            from .core import part,Sketch
            previous=cached[1];parameters=self.router.states[str(layer)].dictionary()
            current=self.make_part(kc,vc if self.router.kind=='V32-Set'else None,parameters['current_weight'],parameters['current_bias'])
            sketch=Sketch(previous.parts[:-1]+[current],previous.costs,previous.origins,previous.history_tokens,
                previous.current_tokens,previous.key_basis,previous.value_basis)
        else:
            sketch=self.make_build(kh,vh,kc,vc,self.router.states[str(layer)].dictionary(),need_value=self.router.kind=='V32-Set')
            if self.history_cache:self._eval_sketch[history_key]=(identity,sketch)
        prepared=self.prepare(sketch,self.router.kind)
        raw_k=torch.cat((kh,kc),1);raw_v=torch.cat((vh,vc),1)
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
        if self.attention_backend=='triton':
            y=sparse_attention(qq,raw_k,raw_v,mask,self.attention_group)
        elif self.attention_backend=='flex':
            from .fast_flex import flex_attention
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


def load_fast(directory,training_seed=0,**options):
    root,m=config(directory)
    torch._dynamo.config.cache_size_limit=128
    torch._dynamo.config.accumulated_cache_size_limit=1024
    router=OnlineRouter(root,m,training_seed).cuda()
    net=FastNetwork(load_teacher('cuda'),router,m['layers'],**options).eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4);net.model_version=m['teacher_sha256']
    return net,m
