"""Frozen inference adapter. Cache only immutable conditioning within one rollout."""
import hashlib
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from streaming_sparse.v08.cache import CacheIdentity,LayerKV
from ..opt20.network import TileNetwork

class RuntimeNetwork(TileNetwork):
    def select(self,name):
        self.runtime_context=None
        for cache in getattr(self,'runtime_cross',[]):cache.clear();cache['is_init']=False
        return super().select(name)

    def forward_chunk(self,latent,embeds,timestep,cache,start,sparse=True,step=0,keep=.2,identity=None,
                      checkpoint=True,collect_aux=True,detach_router=True,mass_only=False,collect_mechanisms=False,write_cache=True):
        from wan.modules.model import sinusoidal_embedding_1d
        if torch.is_grad_enabled():raise RuntimeError("Opt21 inference-only")
        model=self.model
        if collect_mechanisms and torch.is_grad_enabled():raise ValueError('Mechanism probes are evaluation-only')
        if latent.shape[0]!=1 or model.model_type!='t2v':raise ValueError('v08 supports T2V microbatch1')
        device=next(model.parameters()).device;latent=latent.to(device)
        x=model.patch_embedding(latent.permute(0,2,1,3,4))
        grid=torch.tensor([x.shape[2:]],dtype=torch.long)
        x=x.flatten(2).transpose(1,2)
        t=getattr(self,'runtime_timestep',None)
        if t is None:t=torch.full(latent.shape[:2],float(timestep),device=device,dtype=torch.float32)
        features=getattr(self,'runtime_time_features',None)
        if features is None:
            e=model.time_embedding(sinusoidal_embedding_1d(model.freq_dim,t.flatten()).type_as(x))
            e0=model.time_projection(e).unflatten(1,(6,model.dim)).unflatten(0,t.shape)
        else:e,e0=features
        if embeds.shape[0]!=1 or embeds.shape[1]>model.text_len:raise ValueError('Invalid prompt embedding')
        context=model.text_embedding(torch.nn.functional.pad(embeds.to(device),(0,0,0,model.text_len-embeds.shape[1])))
        freqs=getattr(self,'runtime_freqs',None)
        if freqs is None:freqs=model.freqs.to(device)
        if identity is None:
            identity=CacheIdentity(self.model_version,self.router_version,'synthetic','synthetic',0,
                                   start*int(grid[0,1]*grid[0,2]),0.)
        if cache:
            from streaming_sparse.v08.cache import validate
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
            if write_cache:new_cache.append(LayerKV(torch.cat((kh,k),1),torch.cat((vh,v),1),identity,layer,k,v))
            atts.append(la);cals.append(lc);counts.append(count);diagnostics.append(diag)
            mechanism_records.extend(records)
        flow=torch.stack(model.unpatchify(model.head(x,e.unflatten(0,t.shape).unsqueeze(2)),grid)).permute(0,2,1,3,4)
        if getattr(self,'runtime_scheduler',None) is None:
            x0=self.wrapper._convert_flow_pred_to_x0(flow.flatten(0,1),latent.flatten(0,1),t.flatten()).unflatten(0,flow.shape[:2])
        else:
            sigmas,timesteps=self.runtime_scheduler
            timestep_id=torch.argmin((timesteps.unsqueeze(0)-t.flatten().unsqueeze(1)).abs(),dim=1)
            sigma=sigmas[timestep_id].reshape(-1,1,1,1)
            x0=(latent.flatten(0,1).double()-sigma*flow.flatten(0,1).double()).to(flow.dtype).unflatten(0,flow.shape[:2])
        count=torch.stack(counts).sum().clamp_min(1)
        return dict(x0=x0,flow=flow,cache=tuple(new_cache),att_aux=torch.stack(atts).sum()/count,
                    cal_aux=torch.stack(cals).sum()/count,diagnostics=torch.stack(diagnostics).sum(0),mechanisms=mechanism_records)


def configure(net,kernel=False):
    net.__class__=RuntimeNetwork;net.runtime_context=None;net.runtime_cross=[]
    original=net.model.text_embedding.forward
    net.runtime_text_forward=original
    def text_forward(x):
        if net.runtime_context is None:net.runtime_context=original(x)
        return net.runtime_context
    net.model.text_embedding.forward=text_forward
    for block in net.model.blocks:
        cache={'is_init':False};net.runtime_cross.append(cache);fn=block.cross_attn.forward
        def forward(x,context,context_lens,crossattn_cache=None,fn=fn,cache=cache):
            return fn(x,context,context_lens,crossattn_cache=cache)
        block.cross_attn.forward=forward
    if kernel:
        from .pipeline import install
        net.tile_graphs.clear();net.tile_pool=torch.cuda.graph_pool_handle()
        install()
    return net
