"""Check the selected mask and ordered page IDs including ties and signed zero."""
import json
import statistics
import torch

from ..review_v1.ops import Config, RouteOps
from .fixed_page_select import select
from .profile_route import capture
from .selected_id_order import tie_order
from .weights import ROOT


@torch.no_grad()
def main():
    torch.manual_seed(701)
    rows = []
    for pages in (216, 288, 360, 432):
        count = pages // 10
        own = (torch.arange(72, device='cuda') + pages - 72)[None].expand(12, -1).contiguous()

        def reference(scores):
            selection = RouteOps.select(scores, own, Config(), check_valid=False)
            return tie_order(scores, selection['ids'], count), selection['mask']

        def candidate(scores):
            ids, mask = select(scores, own)
            return (tie_order(scores, ids, count) if count <= 32 else ids), mask

        methods = [torch.compile(fn, fullgraph=True, options={'triton.cudagraphs': False})
                   for fn in (reference, candidate)]
        cases = {}
        for spacing in (2, 3, 9, 47):
            scores = torch.full((12, 72, pages), -0., device='cuda')
            scores[..., ::spacing] = 0.
            cases[f'signed_zero_{spacing}'] = scores
        for bins in (1, 2, 20, 4096, 1000000):
            cases[f'bins_{bins}'] = torch.randint(-bins, bins, (12, 72, pages), device='cuda').float()
        errors = {}
        for name, scores in cases.items():
            old, new = (fn(scores) for fn in methods)
            errors[name] = [int((x != y).sum()) for x, y in zip(old, new)]
        graphs = [capture(lambda: fn(scores))[0] for fn in methods]
        times = [[], []]
        for rep in range(30):
            for i in (rep % 2, 1 - rep % 2):
                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(20):
                    graphs[i].replay()
                b.record()
                b.synchronize()
                if rep >= 5:
                    times[i].append(a.elapsed_time(b) / 20)
        ms = list(map(statistics.median, times))
        rows.append(dict(pages=pages, different=errors, ms=ms, speedup=ms[0]/ms[1], samples=times))
        assert all(results == [0, 0] for results in errors.values()), errors
    dest = ROOT / 'artifacts/video_runtime_v1/fixed_page_select_v1'
    dest.mkdir(exist_ok=True)
    (dest/'module_signed.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
