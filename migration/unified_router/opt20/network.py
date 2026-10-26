"""All29-layer video adapter with explicit history updates and static graphs."""
import gc,json
from types import SimpleNamespace
import torch
from .setup import ROOT
from .model import TileRouter
from ..common import config,sha
from ..opt16.pipeline import GroupedRouter
from ..opt19.pipeline import Pipeline
from ..opt9.segmented_network import SegmentedNetwork

class TileGraph:
    def __init__(self,model,q,raw,pool):
        self.q=torch.empty_like(q,memory_format=torch.contiguous_format)
        self.kc=torch.empty_like(raw['kc'],memory_format=torch.contiguous_format)
        self.vc=torch.empty_like(raw['vc'],memory_format=torch.contiguous_format)
        self.pipe=Pipeline(model,raw,32,model.budget,'shared','fp16')
        # Only prototype parts remain needed after grouped history is copied
        # into the persistent KV workspace. Avoid retaining a second full copy.
        self.pipe.cache.history=SimpleNamespace(parts=self.pipe.cache.history.parts)
        self.identity=None;self.q.copy_(q);self.kc.copy_(raw['kc']);self.vc.copy_(raw['vc'])
        def run():
            y,d=self.pipe.run(self.q,self.kc,self.vc,True)
            return y,d['selection']
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):run()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,pool=pool):self.output,self.ids=run()

    def refresh_history(self,kh,vh,identity):
        cache=self.pipe.cache;fresh=cache.builder.history(kh,vh)
        cache.k[:,:cache.nh].copy_(fresh.k);cache.v[:,:cache.nh].copy_(fresh.v)
        # Graphs capture prototype addresses. Copy into existing allocations,
        # never replace the tensor objects on a new trajectory/history.
        for old,new in zip(cache.history.parts,fresh.parts):
            for a,b in zip(old,new):
                if a is not None:a.copy_(b)
        self.pipe.quant.history();self.identity=identity

    def __call__(self,q,raw,identity):
        if self.identity!=identity:self.refresh_history(raw['kh'],raw['vh'],identity)
        self.q.copy_(q);self.kc.copy_(raw['kc']);self.vc.copy_(raw['vc'])
        self.graph.replay();return self.output,self.ids


class TileNetwork(SegmentedNetwork):
    def configure_tile(self,m):
        self.tile_options=m;self.tile_graphs={};self.tile_epoch=0;self.tile_active=None
        self.tile_pool=torch.cuda.graph_pool_handle();self.tile_models=torch.nn.ModuleDict()

    def select(self,name):
        self.tile_epoch+=1
        if name=='Dense':self.set_mode('Dense');return
        variant,stage=name.split('_',1)
        assert variant in ('Tile15','Tile20') and stage in ('before','after')
        if self.tile_active!=name:
            self.tile_graphs.clear();self.tile_models.clear();gc.collect();torch.cuda.empty_cache()
            self.tile_pool=torch.cuda.graph_pool_handle();checks={}
            for layer in self.tile_options['layers']:
                path=ROOT/f'initialization/layer{layer:02d}.pt' if stage=='before' else ROOT/f'training/layer{layer:02d}/{variant}/best.pt'
                state=torch.load(path,map_location='cuda',weights_only=True)
                model=TileRouter(self.tile_options,layer,variant).cuda().eval().requires_grad_(False)
                model.load_state_dict(state['model'])
                model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False)
                self.tile_models[str(layer)]=model;checks[str(path)]=sha(path)
            self.tile_active=name;self.tile_checksums=checks
        self.set_mode('K32-Set');self.router_version='Opt20_'+name

    def _block(self,block,layer,x,e0,grid,freqs,context,history_k,history_v,start,sparse,step,keep,
               collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity):
        if not sparse or layer not in self.layers or not history_k.shape[1] or self.router.kind=='Dense':
            return super()._block(block,layer,x,e0,grid,freqs,context,history_k,history_v,start,False,step,keep,
                collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity)
        if torch.is_grad_enabled() or collect_aux:raise RuntimeError('Opt20 video adapter is evaluation-only')
        from wan.modules.causal_model import causal_rope_apply
        frames=e0.shape[1];tokens=x.shape[1]//frames
        e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2)
        mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2)
        att=block.self_attn;b,s=mod.shape[:2];h=att.num_heads;d=att.head_dim
        q=att.norm_q(att.q(mod)).view(b,s,h,d);k=att.norm_k(att.k(mod)).view(b,s,h,d);v=att.v(mod).view(b,s,h,d)
        q=causal_rope_apply(q,grid,freqs,start_frame=start).type_as(v);k=causal_rope_apply(k,grid,freqs,start_frame=start).type_as(v)
        qq=q[0].transpose(0,1);raw=dict(kh=history_k[0].transpose(0,1),vh=history_v[0].transpose(0,1),kc=k[0].transpose(0,1),vc=v[0].transpose(0,1))
        key=(layer,raw['kh'].shape[1],s)
        if key not in self.tile_graphs:self.tile_graphs[key]=TileGraph(self.tile_models[str(layer)],qq,raw,self.tile_pool)
        identity=(self.tile_epoch,cache_identity,history_k.data_ptr(),history_v.data_ptr(),history_k._version,history_v._version)
        y,ids=self.tile_graphs[key](qq,raw,identity)
        y=y.transpose(0,1)[None]
        x=x+(att.o(y.flatten(2)).unflatten(1,(frames,tokens))*e[2]).flatten(1,2)
        x=x+block.cross_attn(block.norm3(x),context,None,crossattn_cache=None)
        ff=block.ffn((block.norm2(x).unflatten(1,(frames,tokens))*(1+e[4])+e[3]).flatten(1,2))
        x=x+(ff.unflatten(1,(frames,tokens))*e[5]).flatten(1,2)
        nh=raw['kh'].shape[1];nq=h*s
        rows=torch.full((ids.shape[1],),32.,device=ids.device);rows[-1]=s-(len(rows)-1)*32
        cost=torch.where(ids%74==73,8.,64.)*rows[None,:,None]
        rh=torch.where(ids<nh//4680*74,cost,0.).sum();rc=torch.where(ids>=nh//4680*74,cost,0.).sum()
        executed=ids.numel()*32*64;zero=x.new_zeros((),dtype=torch.float32)
        self.routing_records.append(dict(layer=layer,start=start,step=step,queries=nq,history_tokens=nh,current_tokens=s,
            retained_H=rh,retained_C=rc,executed_pairs=executed,keep=(rh+rc)/(nq*(nh+s))))
        diagnostic=torch.stack((rh+rc,rh.new_tensor(nq),rh,rc,rh.new_tensor(nq*(nh+s)),rh.new_tensor(1)))
        return x,k,v,zero,zero,diagnostic,zero,[]

def load():
    _,m=config(ROOT)
    from ..opt14.network import load as load_parent
    net,_=load_parent('streaming_sparse/v10/experiments/opt14_kernels_20261025')
    net.__class__=TileNetwork;net.configure_tile(m);net.layers=set(m['layers'])
    return net,m
