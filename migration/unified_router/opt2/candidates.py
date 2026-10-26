"""Exact candidate selection, skipping fill work only when all slots are valid."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _short_candidates(X, S, O, DIST, HASH, REM, ADD, N: tl.constexpr, B: tl.constexpr, P: tl.constexpr):
    r = tl.program_id(0)
    p = tl.arange(0, P)
    selected = tl.load(S + r * B + p, p < B, False)
    origin = tl.load(O + p, p < B, -1)
    distance = tl.load(DIST + r * B + p, p < B, float('inf'))
    hashes = tl.load(HASH + r * B + p, p < B, float('inf'))
    for side in tl.static_range(2):
        remain = (p < B) & (selected if side == 0 else ~selected)
        result = tl.full((8,), -1, tl.int32)
        cols = tl.arange(0, 8)
        for c in tl.static_range(8):
            allowed = remain
            if c < 3:
                allowed = allowed & (origin == 0)
            elif c < 6:
                allowed = allowed & (origin == 1)
            rank = distance if c < 6 else hashes
            best = tl.min(tl.where(allowed, rank, float('inf')), 0)
            idx = tl.min(tl.where(allowed & (rank == best), p, 2147483647), 0)
            valid = idx < 2147483647
            result = tl.where(cols == c, tl.where(valid, idx, -1), result)
            remain = remain & ~(valid & (p == idx))
        if tl.min(result, 0) < 0:
            for c in tl.static_range(8):
                missing = tl.sum(tl.where(cols == c, result, 0), 0) < 0
                best = tl.min(tl.where(remain, distance, float('inf')), 0)
                idx = tl.min(tl.where(remain & (distance == best), p, 2147483647), 0)
                valid = missing & (idx < 2147483647)
                result = tl.where((cols == c) & valid, idx, result)
                remain = remain & ~(valid & (p == idx))
        if side == 0:
            tl.store(REM + r * 8 + cols, result)
        else:
            tl.store(ADD + r * 8 + cols, result)

def candidate_sides_fast(score, mask, origins, beta, canonical_tile=128, query_stride=1, query_count=None):
    h, n, b = score.shape
    std, mean = torch.std_mean(score, dim=-1, unbiased=False, keepdim=True)
    distance = (score - mean - beta * std).abs()
    q = torch.arange(n, device=score.device) * query_stride
    query_count = n if query_count is None else query_count
    tile_n = torch.minimum(torch.full_like(q, canonical_tile), query_count - q // canonical_tile * canonical_tile)
    row = torch.arange(h, device=score.device)[:, None] * tile_n[None] + (q % canonical_tile)[None]
    hashes = ((torch.arange(b, device=score.device) * 1103515245 + row[..., None] * 12345 + 16) % 2147483647).float()
    rem = torch.empty((h * n, 8), device=score.device, dtype=torch.int64)
    add = torch.empty_like(rem)
    _short_candidates[h * n,](score, mask, origins, distance, hashes, rem, add, h * n, b, tr.next_power_of_2(b), num_warps=4)
    return (rem, add)
