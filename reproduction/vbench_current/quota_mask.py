"""Stable per-head selection for the anchor, history and current partitions."""
import torch
import triton
import triton.language as tl


@triton.jit
def _keep(scores, slots, valid):
    index = tl.arange(0, scores.shape[0])
    ranked = tl.sort(tl.where(valid, scores, -float('inf')), descending=True)
    threshold = tl.sum(tl.where(index == slots - 1, ranked, 0.), 0)
    above = valid & (scores > threshold)
    ties = valid & (scores == threshold)
    remaining = slots - tl.sum(above.to(tl.int32), 0)
    return (slots > 0) & (above | (ties & (tl.cumsum(ties.to(tl.int32), 0) <= remaining)))


@triton.jit
def _mask(R, OWN, CURRENT, MASK, P:tl.constexpr, TOTAL:tl.constexpr, B:tl.constexpr):
    row = tl.program_id(0)
    current = tl.load(CURRENT + row // 72)
    hist = tl.arange(0, B)
    valid = hist < P - 96
    scores = tl.load(R + row*P + 24 + hist, valid, 0.)
    tl.store(MASK + row*P + 24 + hist, _keep(scores, TOTAL-current, valid), valid)
    i = tl.arange(0, 128)
    scores = tl.load(R + row*P + P-72 + i, i < 72, 0.)
    own = tl.load(OWN + row)
    scores = tl.where(i == own-P+72, float('inf'), scores)
    tl.store(MASK + row*P + P-72 + i, _keep(scores, current, i < 72), i < 72)
    tl.store(MASK + row*P + i, True, i < 24)


@torch.library.custom_op('stream_sparse::quota_mask',mutates_args=())
def quota_mask(risk:torch.Tensor, own:torch.Tensor, current:torch.Tensor, total:int)->torch.Tensor:
    h,t,p=risk.shape
    mask=torch.empty((h,t,p),device=risk.device,dtype=torch.bool)
    _mask[(h*t,)](risk.contiguous(),own.contiguous(),current,mask,p,total,triton.next_power_of_2(p-96))
    return mask


@quota_mask.register_fake
def _(risk,own,current,total):
    return torch.empty_like(risk,dtype=torch.bool,memory_format=torch.contiguous_format)
