"""A/B clean commit-time layout generation against deployed CUDA base."""
import ctypes as C
from types import MethodType
from local_experiments.review_v1.runtime import PageStore
import torch

def install(rt):
    rt.set_cuda_fixed15(True,direct_commit=False)
    lib=C.CDLL('/tmp/jzk_commit_tile.so')
    lib.launch.argtypes=[C.c_void_p]*6+[C.c_int]*5+[C.c_void_p]
    attn=rt.packed_attention
    state={'enabled':False,'writes':0}
    def commit(store,k,v,summary,order,start):
        if not state['enabled']:
            return PageStore.commit(store,k,v,summary,order,start)
        if store not in attn.buffers:
            attn.buffers[store]=tuple(k.new_empty((432,12,8192)) for _ in range(2))
        pk,pv=attn.buffers[store]
        assert k.stride(3)==v.stride(3)==1 and all(y.stride(1)==64*y.stride(2) for y in (k,v))
        status=lib.launch(*(x.data_ptr() for x in (k,v,store.cache['k'],store.cache['v'],pk,pv)),store.write,k.stride(0),k.stride(2),v.stride(0),v.stride(2),torch.cuda.current_stream().cuda_stream)
        assert status==0
        end=store.write+72
        if end<=360:
            pages=slice(store.write,end)
            for name in ('keys','mean','detail','stats'):
                getattr(store.summary,name)[:,pages].copy_(getattr(summary,name))
            store.token_ids[:,pages].copy_((order+start).reshape(12,72,64))
        else:
            slots=store.commit_slots(k.device)
            for name in ('keys','mean','detail','stats'):
                getattr(store.summary,name).index_copy_(1,slots,getattr(summary,name))
            store.token_ids.index_copy_(1,slots,(order+start).reshape(12,72,64))
        store.write=24+(end-24)%336
        store.n=min(360,store.n+72)
        store.commits+=1
        attn.history[store]=(store.commits,store.write)
        state['writes']+=1
    for store in rt.stores.values():store.commit=MethodType(commit,store)
    return state
