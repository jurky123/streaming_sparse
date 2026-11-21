"""Transient全窗 paging view. Persist clean KV in the original ring."""
import inspect
import textwrap
import types
import torch
from local_experiments.video_runtime_v1.grouping_dataflow import balanced,group_features
from local_experiments.video_runtime_v1.fixed15_layout import pack_range
from local_experiments.video_runtime_v1.runtime import VideoRuntime

def factors(pages):
    values=[]
    for divisor in (3,2,5):
        while pages%divisor==0:
            values.append(divisor)
            pages//=divisor
    if pages>1:values.append(pages)
    return values

def gather(x,ids):
    return x.gather(1,ids[...,None].expand(-1,-1,x.size(-1)))

def native(x):
    return x.transpose(0,1).unsqueeze(0).contiguous()

def grouped_view(rt,l,model,q,k,v,correct):
    h=12
    pages=k.size(1)//64
    kh=k[0].transpose(0,1)
    vh=v[0].transpose(0,1)
    features=group_features(model,kh.float()[:,None],vh.float()[:,None]).squeeze(1)
    # Rank all visible tokens together, then compact by mandatory provenance.
    ids=balanced(features,factors(pages))
    anchor_end=1536
    current_start=k.size(1)-q.size(1)
    region=(ids>=anchor_end).long()+(ids>=current_start).long()
    ids=ids.gather(1,region.argsort(dim=1,stable=True))
    inside=balanced(gather(features,ids).reshape(h*pages,64,-1),[2,2])
    ids=ids.view(h,pages,64).gather(2,inside.view(h,pages,64)).flatten(1)
    current=ids[:,current_start:]-current_start
    verified=getattr(rt,'verified_window_sizes',set())
    if pages not in verified:
        token=torch.arange(k.size(1),device=k.device)
        assert (ids.sort(1).values==token).all()
        assert (ids[:,:anchor_end]<anchor_end).all() and (current>=0).all()
        assert (ids[:,anchor_end:current_start]<current_start).all()
        # The history pool spans chunks. Record how often pages actually mix them.
        origins=rt.stores[l].token_ids[:,:pages-72].reshape(h,-1)//4608
        origins=origins.gather(1,ids[:,anchor_end:current_start]).reshape(h,-1,64)
        mix=(origins.amax(-1)!=origins.amin(-1)).float().mean().item()
        with open(__file__.replace('whole_window.py','window_audit.log'),'a') as f:
            print(pages,l,mix,file=f)
        verified.add(pages)
        rt.verified_window_sizes=verified
    kg=gather(kh,ids).view(h,pages,64,128)
    vg=gather(vh,ids).view_as(kg)
    # Historical stats use full mode; current pages use the production cheap mode.
    hp=pages-72
    slab=rt.window_slabs[l].slice(0,pages)
    for sl,cheap in [(slice(0,hp),False),(slice(hp,pages),True)]:
        # Some summary kernels assume compact head strides for source tensors.
        s=rt.summary_graph(model,kg[:,sl].contiguous(),vg[:,sl].contiguous(),cheap)
        for name in ('keys','mean','detail','stats'):
            getattr(slab,name)[:,sl].copy_(getattr(s,name))
    en=rt.encode(l,slab) if correct else {}
    qgroup=native(gather(q[0].transpose(0,1),current))
    return qgroup,native(kg.flatten(1,2)),native(vg.flatten(1,2)),slab,en,current

def compute(rt,store,q,k,v,s,basis,ids,workspace):
    # Separate packed buffers keep transient views out of the persistent cache.
    tmp=rt.window_store
    tmp.cache=store.cache
    tmp.sink_pages=store.sink_pages
    tmp.commits+=1
    packed=rt.packed_attention
    pk,pv=packed.cache(tmp)
    pack_range[(k.size(1)//64-72,12,16)](k,v,pk,pv,0)
    return packed(tmp,q,k,v,s,basis,ids,workspace)

def install(rt):
    from local_experiments.review_v1.ops import Summary
    rt.window_store=type('WindowStore',(),dict(packed_writer=True,commits=0,write=0))()
    # Shared scratch is consumed by one layer at a time.
    slab=next(iter(rt.slabs.values()))
    buf=Summary(*(torch.empty_like(getattr(slab,n)) for n in ('keys','mean','detail','stats')))
    rt.window_slabs=dict.fromkeys(rt.slabs,buf)
    src=textwrap.dedent(inspect.getsource(VideoRuntime.forward))
    begin=src.index("        with self.timings.stage('summary_and_encoding'):")
    end=src.index('        direct=self.tile_batch',begin)
    src=src[:begin]+'''        original_q,original_k,original_v=q,kh,vh
        q,kh,vh,s,en,current=grouped_view(self,l,model,q,kh,vh,correct)
'''+src[end:]
    line='mixed=packed(*args,packed_k=True) if packed_k else packed(*args)'
    assert src.count(line)==1
    src=src.replace(line,'mixed=compute(self,*args)')
    src=src.replace('        out=result\n','''        out=native(torch.empty_like(result[0].transpose(0,1)).scatter_(1,current[...,None].expand(-1,-1,128),result[0].transpose(0,1)))
        q,kh,vh=original_q,original_k,original_v
''')
    scope=dict(VideoRuntime.forward.__wrapped__.__globals__,grouped_view=grouped_view,compute=compute,native=native)
    exec(compile(src,__file__,'exec'),scope)
    rt.forward=types.MethodType(scope['forward'],rt)
