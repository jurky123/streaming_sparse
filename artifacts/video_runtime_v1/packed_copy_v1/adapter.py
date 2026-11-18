import ctypes
import torch
from local_experiments.video_runtime_v1.fixed15_attention import PackedAttention
from local_experiments.video_runtime_v1.fixed15_layout import retry,encode,pack_range

class Prepared(PackedAttention):
    def __init__(self):
        super().__init__()
        self.enabled=False
        self.prepare=ctypes.CDLL('/tmp/jzk_prepare.so').run
        self.prepare.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*2+[ctypes.c_void_p]

    def __call__(self, store, q, k, v, summary, basis, ids, workspace):
        if not self.enabled:
            return self.control(store,q,k,v,summary,basis,ids,workspace)
        assert store.packed_writer is not None
        pages = k.shape[1] // 64
        history_pages = pages - 72
        pk, pv = self.cache(store)

        tiles = (pages + 15) // 16
        packed = workspace.tensor('packed_summary', (12, tiles, 10624), q.device, q.dtype)
        flags = workspace.tensor('packed_guard', (12, 72), q.device, torch.int32)
        out = workspace.tensor('packed_output', q.shape, q.device, q.dtype)
        stride = summary.mean.stride(0) // 128
        args = (k,v,pk,pv,summary.keys,summary.mean,summary.detail,packed)
        assert self.prepare(*(x.data_ptr() for x in args), pages, stride,
                            torch.cuda.current_stream(q.device).cuda_stream)==0
        basis = basis.bfloat16().contiguous()
        args = (q, pk, pv, ids, out, packed, basis, flags)
        status = self.kernel(*(x.data_ptr() for x in args), pages, ids.shape[-1],
                             torch.cuda.current_stream(q.device).cuda_stream)
        if status:
            raise RuntimeError(f'fixed15 attention launch returned {status}')
        retry[(72, 12)](flags, q, k, v, ids, summary.keys, summary.mean, summary.detail,
                        basis, out, pages, stride, ids.shape[-1], num_warps=4,
                        num_stages=2, enable_fp_fusion=False)
        self.calls += 1
        return out

    def control(self, store, q, k, v, summary, basis, ids, workspace):
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
        pack(history_pages, pages)
        tiles = (pages + 15) // 16
        packed = workspace.tensor('packed_summary', (12, tiles, 10624), q.device, q.dtype)
        flags = workspace.tensor('packed_guard', (12, 72), q.device, torch.int32)
        out = workspace.tensor('packed_output', q.shape, q.device, q.dtype)
        stride = summary.mean.stride(0) // 128
        encode[(12, tiles, 42)](summary.keys, summary.mean, summary.detail, packed,
                                pages, stride, tiles, num_warps=4)
        basis = basis.bfloat16().contiguous()
        args = (q, pk, pv, ids, out, packed, basis, flags)
        status = self.kernel(*(x.data_ptr() for x in args), pages, ids.shape[-1],
                             torch.cuda.current_stream(q.device).cuda_stream)
        if status:
            raise RuntimeError(f'fixed15 attention launch returned {status}')
        retry[(72, 12)](flags, q, k, v, ids, summary.keys, summary.mean, summary.detail,
                        basis, out, pages, stride, ids.shape[-1], num_warps=4,
                        num_stages=2, enable_fp_fusion=False)
        self.calls += 1
        return out
