"""Compare complete D1536 RMSNorm on saved self/cross attention inputs."""
import json
import statistics
import torch
from .weights import ROOT
from .rms_mean import fused, row_fused

@torch.no_grad()
def main():
    torch.manual_seed(701)
    root = ROOT / 'artifacts/video_runtime_v1'
    (root / 'rms_row_v1').mkdir(exist_ok=True)
    captures = torch.load(root / 'rms_dataflow_v1/input.pt', weights_only=True)
    rows = []
    for name, capture in captures.items():
        x, weight = capture['x'].cuda(), capture['weight'].cuda()
        eps = capture['eps']
        errors = []
        bitwise = []
        for scale in (None, 0., 1e-15, 1e-3, 1., 100., 1e15):
            value = x if scale is None else torch.randn_like(x) * scale
            for epsilon in (eps, 1e-5):
                old = fused(value, weight, epsilon)
                new = row_fused(value, weight, epsilon)
                bitwise.append(torch.equal(old.view(torch.uint8), new.view(torch.uint8)))
                errors.append((old.float() - new.float()).abs().max().item())
        graphs = []
        for fn in (fused, row_fused):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn(x, weight, eps)
            graphs.append((graph, out))
        times = [[], []]
        for rep in range(32):
            for i in (rep % 2, 1 - rep % 2):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(20):
                    graphs[i][0].replay()
                end.record()
                end.synchronize()
                if rep >= 2:
                    times[i].append(start.elapsed_time(end) / 20)
        speedup = statistics.median(times[0]) / statistics.median(times[1])
        rows.append(dict(name=name, shape=list(x.shape), errors=errors, bitwise=bitwise, times_ms=times, speedup=speedup))
        print(name, max(errors), speedup, flush=True)
    (root / 'rms_row_v1/module.json').write_text(json.dumps(rows, indent=2))
    assert all(all(row['bitwise']) for row in rows)

if __name__ == '__main__':
    main()
