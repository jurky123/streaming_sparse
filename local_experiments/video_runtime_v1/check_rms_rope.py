"""Compare the fused and two-stage Q/K transforms before their use in attention."""
import json
import statistics

import torch

from ..review_v1.fused_rope import apply
from .profile_route import capture
from .rms_mean import row_fused, row_rope
from .weights import ROOT


@torch.no_grad()
def main():
    torch.manual_seed(701)
    q, k = (torch.randn(1, 4608, 1536, device='cuda', dtype=torch.bfloat16) for _ in range(2))
    qw, kw = (torch.randn(1536, device='cuda', dtype=torch.bfloat16) for _ in range(2))
    phase = torch.randn(1024, 64, device='cuda', dtype=torch.float64)
    freq = torch.polar(torch.ones_like(phase), phase)
    grid = torch.tensor([[3, 32, 48]])
    rows = []
    for start in (0, 9, 24):
        def separate():
            a, b = (row_fused(x, w, 1e-6).view(1, 4608, 12, 128) for x, w in ((q, qw), (k, kw)))
            return apply(a, b, grid, freq, start)

        def fused():
            return tuple(row_rope(x, w, 1e-6, freq, 32, 48, start) for x, w in ((q, qw), (k, kw)))

        variants = [capture(fn) for fn in (separate, fused)]
        diff = [int((a.view(torch.int16) != b.view(torch.int16)).sum())
                for a, b in zip(variants[0][1], variants[1][1])]
        times = [[], []]
        for rep in range(25):
            for i in (rep % 2, 1 - rep % 2):
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin.record()
                for _ in range(16):
                    variants[i][0].replay()
                end.record()
                end.synchronize()
                if rep >= 3:
                    times[i].append(begin.elapsed_time(end) / 16)
        rows.append(dict(start=start, diff=diff, ms=list(map(statistics.median, times)), samples=times))
    path = ROOT / 'artifacts/video_runtime_v1/rms_rope_v1'
    path.mkdir(exist_ok=True)
    (path / 'module.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
