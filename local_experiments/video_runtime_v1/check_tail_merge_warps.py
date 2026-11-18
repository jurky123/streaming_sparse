"""Check occupancy tradeoffs with the deployed merged tail and real exact output."""
import json
import statistics

import torch
import triton

from ..gap_exact_round_kernel import _attn_fwd
from ..high_sparse_runtime import _budget_mask
from ..review_v1.tuned_attention import _tail
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
    for p in (216, 288, 360, 432):
        count = p // 10
        ids = (data['ids'][..., :count] % p).cuda().contiguous()[None]
        keys, mean, detail = (data['summary'][name][:, :p].cuda().contiguous()
                              for name in ('keys', 'mean', 'detail'))
        exact = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty(h, n, device=q.device)
        bits = torch.empty(h, n // 64, 16, device=q.device, dtype=torch.uint32)
        _attn_fwd[(n//64,h)](q,k,v,d**-.5,count,ids,lse,exact,h,n,p*64,n//64,d,
                             64,64,HIGH_PRECISION_AV=True,num_warps=4,num_stages=2)
        _budget_mask[(n//64,h)](ids,bits,n//64,count,triton.next_power_of_2(count))
        variants = []
        for warps in (4, 8):
            out = torch.empty_like(q)
            def run():
                return _tail[(n//64,h)](q,keys,mean,detail,bits,None,None,None,p,n,h,d,
                    BQ=64,BR=64,BF16_AV=True,MERGE=True,EXACT=exact,ELSE=lse,BASIS=basis,
                    OUT=out,num_warps=warps,num_stages=2,enable_fp_fusion=False)
            compiled = run()
            graph, _ = capture(run)
            variants.append((graph, out, compiled))
        times = [[], []]
        for rep in range(26):
            for idx in (rep%2, 1-rep%2):
                a,b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(12):
                    variants[idx][0].replay()
                b.record()
                b.synchronize()
                if rep >= 2:
                    times[idx].append(a.elapsed_time(b)/12)
        old,new = (x[1] for x in variants)
        row = dict(pages=p,ms=list(map(statistics.median,times)),different=int((old!=new).sum()),
                   max_abs=float((old.float()-new.float()).abs().max()),
                   registers=[v[2].n_regs for v in variants],spills=[v[2].n_spills for v in variants],
                   shared=[v[2].metadata.shared for v in variants])
        print(row, flush=True)
        rows.append(dict(**row,samples_ms=times))
    dest = root / 'tail_merge_warps_v1'
    dest.mkdir(exist_ok=True)
    (dest/'module.json').write_text(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
