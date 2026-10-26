"""Execution-only acceleration; preserve original router parameters and math.

Replay the original 64-iteration budget solve as a CUDA graph. No compiler,
fusion, reduced iteration count, changed dtype, or changed gradient formula.
Returned probabilities own their storage, including during checkpoint replay.
"""
import hashlib
from pathlib import Path
import torch

_graphs = {}
_original = None
stats = dict(captures=0, replays=0)
_old_batches = None


def identity():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def solve(scores, costs, active, budget, temperature):
    lo = scores.masked_fill(~active, torch.inf).amin(-1, keepdim=True)-40*temperature
    hi = scores.masked_fill(~active, -torch.inf).amax(-1, keepdim=True)+40*temperature
    for _ in range(64):
        mid = (lo+hi)/2
        mass = (costs*torch.sigmoid((scores-mid)/temperature)*active).sum(-1, keepdim=True)
        above = mass > budget
        lo, hi = torch.where(above, mid, lo), torch.where(above, hi, mid)
    return torch.sigmoid((scores-(lo+hi)/2)/temperature)*active


def forward(ctx, scores, costs, budget, temperature):
    active = costs.detach() > 0
    interior = 0 < budget < float(costs.detach().sum())
    if not interior or scores.device.type != 'cuda':
        return _original(ctx, scores, costs, budget, temperature)
    # All cost values are copied on every invocation, so this cache is valid for
    # changing metadata as well as the three fixed replay history lengths.
    key = (scores.device.index, scores.dtype, tuple(scores.shape), costs.dtype,
           tuple(costs.shape), int(budget), float(temperature))
    if key not in _graphs:
        si = scores.detach().contiguous().clone()
        ci = costs.detach().contiguous().clone()
        ai = active.clone()
        stream = torch.cuda.Stream(device=scores.device)
        stream.wait_stream(torch.cuda.current_stream(scores.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                p = solve(si, ci, ai, budget, temperature)
        torch.cuda.current_stream(scores.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            p = solve(si, ci, ai, budget, temperature)
        torch.cuda.current_stream(scores.device).wait_stream(stream)
        _graphs[key] = (si, ci, ai, graph, p)
        stats['captures'] += 1
    si, ci, ai, graph, stored = _graphs[key]
    si.copy_(scores); ci.copy_(costs); ai.copy_(active)
    graph.replay()
    p = stored.clone()
    stats['replays'] += 1
    ctx.save_for_backward(p, costs)
    ctx.temperature, ctx.interior = temperature, interior
    return p


def install():
    global _original, _old_batches
    from streaming_sparse.v05.primitives import BudgetGate
    from streaming_sparse.v08.execution import accelerate
    if _original is None:
        _original = BudgetGate.forward
        BudgetGate.forward = staticmethod(forward)
        _old_batches = (accelerate.SCORE_BATCH, accelerate.GATE_BATCH)
        accelerate.SCORE_BATCH = 512
        accelerate.GATE_BATCH = 256


def uninstall():
    global _original, _old_batches
    from streaming_sparse.v05.primitives import BudgetGate
    from streaming_sparse.v08.execution import accelerate
    if _original is not None:
        BudgetGate.forward = staticmethod(_original)
        _original = None
        accelerate.SCORE_BATCH, accelerate.GATE_BATCH = _old_batches

