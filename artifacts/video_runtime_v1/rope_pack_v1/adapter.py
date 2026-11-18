"""Switch just the sparse K producer and the corresponding prepare kernels."""
import ctypes
from types import SimpleNamespace

import torch

from local_experiments.video_runtime_v1 import fixed15_attention as attention
from local_experiments.video_runtime_v1 import rms_mean


class Switch:
    def __init__(self, runtime):
        self.runtime = runtime
        runtime.packed_attention.pack_rotary_keys = False
        self.original = attention.library()
        self.vonly = ctypes.CDLL('/tmp/jzk_ropepack_vonly.so')
        for name in ('prepare_fixed15', 'prepare_current_fixed15'):
            getattr(self.vonly, name).argtypes = getattr(self.original, name).argtypes
        self.dual = ctypes.CDLL('/tmp/jzk_ropepack_shared.so').rms_rope
        self.dual.argtypes = rms_mean.library().rms_rope.argtypes + [ctypes.c_void_p]
        self.dual.restype = None
        self.proxy = SimpleNamespace(commit_fixed15=self.original.commit_fixed15, run=self.original.run)
        attention.library = lambda: self.proxy
        self.blocks = []
        for layer, store in runtime.stores.items():
            module = runtime.engine.wrapper.model.blocks[layer].self_attn
            assert module.project_cached_rotary is rms_mean.project_cached
            self.blocks.append((module, self.projection(store)))

    def projection(self, store):
        def project(module, x, grid, freq, start, key_out, value_out):
            if store.n <= 72 or self.runtime.engine.current['step'] == 4 or not self.runtime.sparse:
                return rms_mean.project_cached(module,x,grid,freq,start,key_out,value_out)
            _, gh, gw = grid[0].tolist()
            q = rms_mean.row_rope(module.q(x),module.norm_q.weight,module.norm_q.eps,freq,gh,gw,start)
            raw = module.k(x)
            packed = self.runtime.packed_attention.cache(store)[0]
            self.dual(raw.data_ptr(),module.norm_k.weight.data_ptr(),key_out.data_ptr(),4608,
                      module.norm_k.eps,freq.data_ptr(),gh,gw,start,packed[store.n].data_ptr(),
                      torch.cuda.current_stream(x.device).cuda_stream)
            torch.addmm(module.v.bias,x.view(-1,1536),module.v.weight.T,out=value_out.view(-1,1536))
            self.calls += 1
            return q,key_out,value_out
        return project

    def select(self, candidate):
        self.calls = 0
        lib = self.vonly if candidate else self.original
        self.proxy.prepare_fixed15 = lib.prepare_fixed15
        self.proxy.prepare_current_fixed15 = lib.prepare_current_fixed15
        self.runtime.packed_attention.prepare_kernel = lib.prepare_fixed15
        for module, project in self.blocks:
            module.project_cached_rotary = project if candidate else rms_mean.project_cached
