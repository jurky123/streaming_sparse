"""Compare both fused residual outputs with the deployed BF16 operations."""
import argparse
import json
import statistics

import torch

from .block_pointwise import fused
from .layernorm_mod import apply, add_and_modulate, gated_and_normalize
from .profile_route import capture
from .weights import ROOT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gated', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(701)
    norm = torch.nn.LayerNorm(1536, eps=1e-6, device='cuda', dtype=torch.bfloat16)
    norm.weight.normal_()
    norm.bias.normal_()
    rows = []
    for factor in (0.01, 1., 16.):
        x, y = (torch.randn(1, 4608, 1536, device='cuda', dtype=torch.bfloat16)*factor for _ in range(2))
        shift, scale = torch.randn(1, 3, 6, 1536, device='cuda', dtype=x.dtype).chunk(6, dim=2)[3:5]

        def baseline():
            if args.gated:
                residual = fused(x, y, scale, None, False)
                return residual, norm(residual)
            residual = x + y
            return residual, apply(residual, scale, shift, norm.eps)

        def candidate():
            if args.gated:
                return gated_and_normalize(x, y, scale, norm)
            return add_and_modulate(x, y, scale, shift, norm.eps)

        variants = [capture(fn) for fn in (baseline, candidate)]
        samples = [[], []]
        for rep in range(32):
            for i in (rep%2, 1-rep%2):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(40):
                    variants[i][0].replay()
                end.record()
                end.synchronize()
                if rep >= 2:
                    samples[i].append(start.elapsed_time(end)/40)
        ms = list(map(statistics.median, samples))
        equal = [torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a,b in zip(variants[0][1],variants[1][1])]
        assert all(equal)
        rows.append(dict(scale=factor, ms=ms, speedup=ms[0]/ms[1], equal=equal, samples_ms=samples))
    path = ROOT/'artifacts/video_runtime_v1/residual_norm_v1'
    path.mkdir(exist_ok=True)
    (path/('gated.json' if args.gated else 'module.json')).write_text(json.dumps(rows,indent=2)+'\n')


if __name__ == '__main__':
    main()
