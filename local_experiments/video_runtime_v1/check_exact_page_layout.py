"""Isolate the cost of reading one selected KV page with different contiguous tiles."""
import argparse
import json
import statistics

import torch

from ..gap_exact_round_kernel import _attn_fwd
from .profile_route import capture
from .weights import ROOT


def pack(x, width):
    pages = x.view(-1, 64, 12, 128 // width, width)
    return pages.permute(0, 2, 3, 1, 4).contiguous()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True, choices=(1, 8, 16, 32, 128))
    parser.add_argument('--page-v', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'artifacts/video_runtime_v1'
    data = torch.load(root / 'tail_liveness_v1/input.pt', weights_only=True)
    q, k, v = (data[key].cuda().contiguous() for key in ('q', 'k', 'v'))
    kp = pack(k, args.width)
    vp = pack(v, 128) if args.page_v else v
    results = []
    for pages in (216, 288, 360, 432):
        count = int(.1 * pages)
        ids = (data['ids'][..., :count].cuda() % pages).contiguous()
        graphs, outputs, resources = [], [], []
        for key, value, tile, page_v in ((k, v, 0, False), (kp, vp, args.width, args.page_v)):
            out = torch.empty_like(q, dtype=torch.float32)
            lse = torch.empty(12, 4608, device=q.device)

            def call():
                return _attn_fwd[(72, 12)](q, key, value, 128**-.5, count, ids, lse, out, 12,
                                           4608, pages * 64, 72, 128, 64, 64, HIGH_PRECISION_AV=True,
                                           PAGE_K_TILE=tile, PAGE_V=page_v, num_warps=4, num_stages=2)
            kernel = call()
            resources.append([kernel.n_regs, kernel.n_spills, kernel.metadata.shared])
            graph, _ = capture(call)
            graphs.append(graph)
            outputs.append((out, lse))
        samples = [[], []]
        for rep in range(32):
            for i in (rep % 2, 1 - rep % 2):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(10):
                    graphs[i].replay()
                end.record()
                end.synchronize()
                if rep >= 2:
                    samples[i].append(start.elapsed_time(end) / 10)
        ms = list(map(statistics.median, samples))
        equal = [torch.equal(a, b) for a, b in zip(*outputs)]
        row = dict(pages=pages, ms=ms, speedup=ms[0]/ms[1], equal=equal, resources=resources)
        print(row, flush=True)
        results.append(dict(**row, samples_ms=samples))
    dest = root / 'exact_page_layout_v1'
    dest.mkdir(exist_ok=True)
    (dest / f'k{args.width}_v{int(args.page_v)}.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
