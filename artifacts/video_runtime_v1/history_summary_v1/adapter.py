"""Per-layer packed summary cache for eager base acceptance tests."""
import ctypes
import torch
from local_experiments.video_runtime_v1.fixed15_attention import PackedAttention

class SummaryWorkspace:
    def __init__(self,shared,packed):
        self.shared=shared
        self.packed=packed

    def tensor(self,name,shape,device,dtype):
        if name=='packed_summary':
            return self.packed
        return self.shared.tensor(name,shape,device,dtype)

class CachedAttention(PackedAttention):
    def __init__(self):
        super().__init__()
        self.summaries={}
        self.versions={}
        self.full=ctypes.CDLL('/tmp/jzk_compact_half.so').prepare_fixed15
        self.full.argtypes=self.prepare_kernel.argtypes
        self.prepare_kernel=self.full
        self.partial=ctypes.CDLL('/tmp/jzk_history_summary.so').prepare_fixed15
        self.partial.argtypes=self.full.argtypes
        self.hits=0
        self.enabled=True

    def reset(self):
        super().reset()
        self.versions.clear()
        self.hits=0

    def __call__(self,store,q,k,v,summary,basis,ids,workspace):
        if not self.enabled or torch.cuda.is_current_stream_capturing():
            self.versions.pop(store,None)
            return super().__call__(store,q,k,v,summary,basis,ids,workspace)
        pages=k.shape[1]//64
        tiles=(pages+15)//16
        if store not in self.summaries:
            self.summaries[store]=q.new_empty(12*27*10624)
        packed=self.summaries[store][:12*tiles*10624].view(12,tiles,10624)
        version=(store.commits,pages)
        hit=self.versions.get(store)==version
        self.prepare_kernel=self.partial if hit else self.full
        try:
            result=super().__call__(store,q,k,v,summary,basis,ids,SummaryWorkspace(workspace,packed))
        finally:
            self.prepare_kernel=self.full
        self.hits+=hit
        self.versions[store]=version
        return result

def install(rt):
    rt.packed_attention=CachedAttention()
    for store in rt.stores.values():
        store.packed_writer=rt.packed_attention.commit
