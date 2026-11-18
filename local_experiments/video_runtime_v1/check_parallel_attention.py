"""Compare the deployed merged tail to concurrent exact and compensation streams."""
import json
import statistics
import torch
import triton

from ..gap_exact_round_kernel import _attn_fwd
from ..high_sparse_runtime import _budget_mask
from ..review_v1.tuned_attention import _merge, _tail
from .profile_route import capture
from .weights import ROOT


@torch.no_grad()
def main():
    root = ROOT / 'artifacts/video_runtime_v1'
    data = torch.load(root / 'tail_liveness_v1/input.pt', weights_only=True)
    q, k, v = (data[name].cuda().contiguous() for name in ('q', 'k', 'v'))
    basis = data['basis'].cuda().bfloat16().contiguous()
    _, n, h, d = q.shape
    rows = []
    side = torch.cuda.Stream()
    for p in (216, 288, 360, 432):
        count = p // 10
        ids = (data['ids'][..., :count] % p).cuda().contiguous()[None]
        keys, mean, detail = (data['summary'][name][:, :p].cuda().contiguous()
                              for name in ('keys', 'mean', 'detail'))
        exact = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty(h, n, device=q.device)
        bits = torch.empty(h, n // 64, 16, device=q.device, dtype=torch.uint32)
        mu = torch.empty((h, n, d), device=q.device)
        code = torch.empty((h, n, 8), device=q.device)
        lt = torch.empty_like(lse)
        variants = []
        for concurrent in (False, True):
            out = torch.empty_like(q)

            def evaluate():
                main_stream = torch.cuda.current_stream()
                if concurrent:
                    side.wait_stream(main_stream)
                _attn_fwd[(n//64,h)](q,k,v,d**-.5,count,ids,lse,exact,h,n,p*64,n//64,d,
                                     64,64,HIGH_PRECISION_AV=True,num_warps=4,num_stages=2)
                with torch.cuda.stream(side if concurrent else main_stream):
                    _budget_mask[(n//64,h)](ids,bits,n//64,count,triton.next_power_of_2(count),num_warps=4)
                    _tail[(n//64,h)](q,keys,mean,detail,bits,mu,code,lt,p,n,h,d,
                        BF16_AV=True,MERGE=not concurrent,EXACT=exact,ELSE=lse,BASIS=basis,
                        OUT=out,num_warps=4,num_stages=2,enable_fp_fusion=False)
                if concurrent:
                    main_stream.wait_stream(side)
                    _merge[(n//16,h)](exact,lse,mu,code,lt,basis,out,n,h,d,num_warps=4,enable_fp_fusion=False)
                return out
            variants.append(capture(evaluate))
        times = [[], []]
        for rep in range(26):
            for i in (rep % 2, 1 - rep % 2):
                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(12):
                    variants[i][0].replay()
                b.record()
                b.synchronize()
                if rep >= 2:
                    times[i].append(a.elapsed_time(b) / 12)
        old, new = (item[1] for item in variants)
        ms = list(map(statistics.median, times))
        row = dict(pages=p, different=int((old.view(torch.int16)!=new.view(torch.int16)).sum()),
                   max_abs=float((old.float()-new.float()).abs().max()), ms=ms, speedup=ms[0]/ms[1])
        print(row, flush=True)
        rows.append(dict(**row,samples_ms=times))
    dest = root / 'parallel_attention_v1'
    dest.mkdir(exist_ok=True)
    (dest/'module.json').write_text(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
