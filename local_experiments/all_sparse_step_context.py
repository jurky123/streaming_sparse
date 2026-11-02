"""Checkpoint each denoising forward with its own immutable engine context."""
import types,torch
from torch.utils.checkpoint import checkpoint

def install_step_context(rt):
    engine=rt.engine
    # Layer 0 uses the upstream in-place KV writer. Give each grad forward
    # private detached cache storage; preserve input gradients through new KV.
    original_attention=rt.original[0]
    def isolated_attention(module,*args,**kwargs):
        if rt.training_event and torch.is_grad_enabled() and kwargs.get('kv_cache') is not None:
            kwargs=dict(kwargs)
            kwargs['kv_cache']={k:v.detach().clone() if torch.is_tensor(v) else v for k,v in kwargs['kv_cache'].items()}
        return original_attention(*args,**kwargs)
    engine.wrapper.model.blocks[0].self_attn.forward=types.MethodType(isolated_attention,engine.wrapper.model.blocks[0].self_attn)
    for i,b in enumerate(engine.wrapper.model.blocks):
        original=rt.original_blocks[i]
        def wrapped(module,*args,index=i,original=original,**kwargs):
            if not(rt.training_event and index>0 and torch.is_grad_enabled()):return original(*args,**kwargs)
            context=dict(engine.current)
            def replay(*xs,**kw):
                previous=dict(engine.current);engine.current.update(context)
                try:return original(*xs,**kw)
                finally:engine.current.clear();engine.current.update(previous)
            return checkpoint(replay,*args,**kwargs,use_reentrant=False)
        b.forward=types.MethodType(wrapped,b)
