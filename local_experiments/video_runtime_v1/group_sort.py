"""Stable ascending score sort and propagation of token ids for small groups."""
import torch
import triton
import triton.language as tl


@triton.jit
def _order(score, row):
    bits = tl.where(score == 0., 0., score).to(tl.uint32, bitcast=True)
    key = bits ^ tl.where((bits >> 31) != 0, 0xffffffff, 0x80000000).to(tl.uint32)
    key = tl.where(score != score, 0xffffffff, key).to(tl.uint32)
    return tl.sort((key.to(tl.uint64) << 32) | row.to(tl.uint64)).to(tl.int32)


@triton.jit
def _large_order(S, ORDER, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    score = tl.load(S + group * SIZE + row, row < SIZE, float("nan"))
    tl.store(ORDER + group * SIZE + row, _order(score, row), row < SIZE)


@triton.jit
def _gather_rows(ORDER, X, IDS, Y, OUT, SIZE: tl.constexpr):
    group = tl.program_id(1)
    row = tl.program_id(0) * 64 + tl.arange(0, 64)
    col = tl.arange(0, 32)
    base = group * SIZE
    source = base + tl.load(ORDER + base + row)
    tl.store(OUT + base + row, tl.load(IDS + source))
    tl.store(Y + (base + row[:,None]) * 32 + col[None,:],
             tl.load(X + source[:,None] * 32 + col[None,:]))


@triton.jit
def _sort_rows(S, X, IDS, Y, OUT, SIZE: tl.constexpr, COPY_X: tl.constexpr):
    group = tl.program_id(0)
    row = tl.arange(0, SIZE)
    score = tl.load(S + group * SIZE + row)
    order = _order(score, row)
    ids = tl.load(IDS + group * SIZE + order)
    tl.store(OUT + group * SIZE + row, ids)
    if COPY_X:
        col = tl.arange(0, 32)
        values = tl.load(X + (group * SIZE + order[:, None]) * 32 + col[None, :])
        tl.store(Y + (group * SIZE + row[:, None]) * 32 + col[None, :], values)


def propagate(score, x, ids, copy_features, radix=False):
    groups, size = score.shape
    out = torch.empty_like(ids)
    y = torch.empty_like(x) if copy_features else x
    if size in (1536, 4608):
        order = torch.empty(score.shape, dtype=torch.int32, device=score.device)
        if radix:
            from .group_radix import order as radix_order
            radix_order(score, order)
        else:
            _large_order[(groups,)](score, order, size, triton.next_power_of_2(size), num_warps=8)
        _gather_rows[(size // 64, groups)](order, x, ids, y, out, size)
        return y, out
    _sort_rows[(groups,)](score, x, ids, y, out, size, copy_features, num_warps=4)
    return y, out
