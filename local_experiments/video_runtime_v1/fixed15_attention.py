"""Packed SM80 attention for the fixed 15% base."""
import ctypes
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

from .fixed15_layout import pack_range, retry
import torch
from filelock import FileLock


@lru_cache(None)
def library():
    source=Path(__file__).parent/'csrc/fixed15_attention.cu'
    folder=Path('/tmp/streaming_sparse-0/fixed15_attention')/hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder.mkdir(exist_ok=True,parents=True)
    binary=folder/'kernel.so'
    with FileLock(str(folder/'build.lock')):
        if not binary.exists():
            tmp=binary.with_suffix('.tmp.so')
            subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80',
                            '--shared','-Xcompiler','-fPIC',str(source),'-o',str(tmp)],check=True)
            tmp.replace(binary)
    lib=ctypes.CDLL(str(binary))
    lib.run.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    lib.commit_fixed15.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    lib.prepare_fixed15.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    lib.prepare_current_fixed15.argtypes = lib.prepare_fixed15.argtypes
    lib.prepare_v_fixed15.argtypes = lib.prepare_fixed15.argtypes[:-1] + [ctypes.c_bool, ctypes.c_void_p]
    return lib


class PackedAttention:
    def __init__(self):
        self.pack_rotary_keys = True
        self.buffers = {}
        self.history = {}
        self.packed_summaries = {}
        self.summary_versions = {}
        self.summary_hits = 0
        self.captured = False
        self.calls = 0
        self.direct_commits = 0
        self.kernel = library().run
        self.prepare_kernel = library().prepare_fixed15

    def accepts_packed_key(self):
        return self.pack_rotary_keys and self.prepare_kernel is library().prepare_fixed15

    def reset(self):
        self.history.clear()
        self.summary_versions.clear()
        self.summary_hits = 0
        self.calls = 0
        self.direct_commits = 0

    def cache(self, store):
        if store not in self.buffers:
            src = store.cache['k']
            self.buffers[store] = tuple(src.new_empty((432,12,8192)) for _ in range(2))
        return self.buffers[store]

    def commit(self, store, k, v):
        # Grouping emits [head,page,token,dim] with contiguous page rows.
        pk, pv = self.cache(store)
        args = (k, v, store.cache['k'], store.cache['v'], pk, pv)
        status = library().commit_fixed15(*(x.data_ptr() for x in args), store.write,
                                          k.stride(0), k.stride(2), v.stride(0), v.stride(2),
                                          torch.cuda.current_stream(k.device).cuda_stream)
        if status:
            raise RuntimeError(f'fixed15 commit launch returned {status}')
        self.direct_commits += 1

    def __call__(self, store, q, k, v, summary, basis, ids, workspace, packed_k=False):
        pages = k.shape[1] // 64
        history_pages = pages - 72
        pk, pv = self.cache(store)

        def pack(start, end):
            if start < end:
                pack_range[(end-start, 12, 16)](k, v, pk, pv, start)

        if getattr(store, 'packed_writer', None) is None:
            previous = self.history.get(store)
            if torch.cuda.is_current_stream_capturing():
                # Replays do not execute the Python commit-counter check.
                pack(0, history_pages)
            elif previous is None or store.commits != previous[0]:
                if previous is not None and store.commits == previous[0] + 1:
                    start = previous[1]
                    end = min(start + 72, 360)
                    pack(start, end)
                    pack(24, 24 + 72 - (end-start))
                else:
                    pack(0, history_pages)
            self.history[store] = store.commits, store.write
        tiles = (pages + 15) // 16
        shape = (12, tiles, 10624)
        prepare = self.prepare_kernel
        # Captured replays can change the history without executing Python.
        if not self.captured and torch.cuda.is_current_stream_capturing():
            self.captured = True
            self.summary_versions.clear()
        cacheable = not self.captured and prepare is library().prepare_fixed15
        if cacheable:
            if store not in self.packed_summaries:
                self.packed_summaries[store] = q.new_empty(12*27*10624)
            packed = self.packed_summaries[store][:12*tiles*10624].view(shape)
            version = (store.commits, pages)
            if self.summary_versions.get(store) == version:
                prepare = library().prepare_current_fixed15
                self.summary_hits += 1
        else:
            self.summary_versions.pop(store, None)
            packed = workspace.tensor('packed_summary', shape, q.device, q.dtype)
        flags = workspace.tensor('packed_guard', (12, 72), q.device, torch.int32)
        out = workspace.tensor('packed_output', q.shape, q.device, q.dtype)
        stride = summary.mean.stride(0) // 128
        inputs = (k, v, pk, pv, summary.keys, summary.mean, summary.detail, packed)
        stream = torch.cuda.current_stream(q.device).cuda_stream
        extra = ()
        if packed_k:
            extra = (prepare is library().prepare_current_fixed15,)
            prepare = library().prepare_v_fixed15
        status = prepare(*(x.data_ptr() for x in inputs), pages, stride, *extra, stream)
        if status:
            raise RuntimeError(f'fixed15 layout preparation returned {status}')
        if cacheable:
            self.summary_versions[store] = version
        basis = basis.bfloat16().contiguous()
        args = (q, pk, pv, ids, out, packed, basis, flags)
        status = self.kernel(*(x.data_ptr() for x in args), pages, ids.shape[-1], stream)
        if status:
            raise RuntimeError(f'fixed15 attention launch returned {status}')
        retry[(72, 12)](flags, q, k, v, ids, summary.keys, summary.mean, summary.detail,
                        basis, out, pages, stride, ids.shape[-1], num_warps=4,
                        num_stages=2, enable_fp_fusion=False)
        self.calls += 1
        return out
