"""Measure stable id and feature propagation at all power-of-two grouping levels."""
import argparse
import json
import statistics
import torch
from .group_sort import propagate
from .weights import ROOT
from .profile_route import capture


def baseline(s, x, ids, copy_features):
    order = s.argsort(dim=-1, stable=True)
    y = x.gather(1, order[:, :, None].expand_as(x)) if copy_features else x
    return y, ids.gather(1, order)


def main():
    torch.manual_seed(701)
    parser = argparse.ArgumentParser()
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--radix',action='store_true')
    args=parser.parse_args()
    large=args.large or args.radix
    result = []
    for size, copy in (((1536, True),(4608, True)) if large else ((512, True), (256, True), (128, False), (64, True), (32, False))):
        groups = 12*4608//size
        x = torch.randn(groups, size, 32, device='cuda')
        s = x[:, :, 0].contiguous()
        s[:, ::5] = -0.
        s[:, 1::5] = 0.
        s[:, 2::17] = float('nan')
        ids = torch.arange(groups*size, device='cuda').reshape(groups, size).roll(7, 1)
        cases = [capture(lambda:baseline(s,x,ids,copy)),capture(lambda:propagate(s,x,ids,copy,radix=args.radix))]
        times = [[], []]
        for r in range(32):
            for i in (r%2, 1-r%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(20): cases[i][0].replay()
                b.record()
                b.synchronize()
                if r>=2: times[i].append(a.elapsed_time(b)/20)
        equal = [torch.equal(a,b) for a,b in zip(cases[0][1],cases[1][1])]
        ms=list(map(statistics.median,times))
        row=dict(size=size,copy=copy,equal=equal,ms=ms,speedup=ms[0]/ms[1])
        print(row,flush=True)
        result.append(dict(**row,samples_ms=times))
    dst=ROOT/'artifacts/video_runtime_v1/group_sort_v1'
    dst.mkdir(exist_ok=True)
    (dst/('radix_module.json' if args.radix else 'large_module.json' if large else 'module.json')).write_text(json.dumps(result,indent=2))
    assert all(all(r['equal']) for r in result)


if __name__=='__main__':
    main()
