"""Paired QK graph timings across all deployed page counts."""
import argparse
import json
import math
import statistics
import torch
from . import tensor_bmm
from .profile_route import capture
from .weights import ROOT

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale-in-qk', action='store_true')
    args = parser.parse_args()
    folder = ROOT / 'artifacts/video_runtime_v1'
    data = torch.load(folder / 'tail_liveness_v1/input.pt', weights_only=True)
    index = torch.linspace(0, 63, 16, device='cuda').round().long()
    probes = data['q'].cuda()[0].transpose(0, 1).reshape(12, 72, 64, 128).index_select(2, index).flatten(1, 2)
    tensor_bmm.QK_TILE64 = True
    @torch.compile(fullgraph=True, dynamic=False)
    def old(q, k):
        return tensor_bmm.bf16_inputs_bmm(q.float(), k.float()) / math.sqrt(128)
    @torch.compile(fullgraph=True, dynamic=False)
    def native(q, k):
        return tensor_bmm.bf16_inputs_bmm(q, k) / math.sqrt(128)
    results = []
    for pages in (216, 288, 360, 432):
        keys = data['summary']['keys'][:, :pages].cuda().flatten(1, 2).transpose(1, 2)
        variants = [capture(lambda: fn(probes, keys)) for fn in (old, tensor_bmm.scaled_route_qk if args.scale_in_qk else native)]
        times = [[], []]
        for rep in range(32):
            for i in (rep % 2, 1 - rep % 2):
                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(20):
                    variants[i][0].replay()
                b.record()
                b.synchronize()
                if rep >= 2:
                    times[i].append(a.elapsed_time(b) / 20)
        a, b = [v[1] for v in variants]
        equal = torch.equal(a.view(torch.int32), b.view(torch.int32))
        speedup = statistics.median(times[0]) / statistics.median(times[1])
        results.append(dict(pages=pages, equal=equal, max_abs=(a-b).abs().max().item(),
                            speedup=speedup, ms=times))
        print(pages, equal, speedup, flush=True)
    target = folder / ('scaled_route_qk_v1' if args.scale_in_qk else 'native_route_qk_v1')
    target.mkdir(exist_ok=True)
    (target / 'module.json').write_text(json.dumps(results, indent=2))
    assert all(row['equal'] for row in results)

if __name__ == '__main__':
    main()
