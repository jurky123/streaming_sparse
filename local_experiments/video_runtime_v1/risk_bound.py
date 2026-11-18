"""Compute the per-page value-norm bound from representative summary statistics."""
import torch
import triton
import triton.language as tl


@triton.jit
def _bound(ST, C, P:tl.constexpr, B:tl.constexpr):
    h = tl.program_id(0)
    p = tl.arange(0, B)
    bp = tl.load(ST + ((h*P+p)*4+1)*12+11, p<P, -float('inf')).to(tl.float32)
    tl.store(C+h*P+p, bp+tl.max(bp,0), p<P)


@torch.library.custom_op('streaming_sparse::risk_bound',mutates_args=())
def bound(stats:torch.Tensor)->torch.Tensor:
    h,p = stats.shape[:2]
    c = torch.empty((h,p), device=stats.device, dtype=torch.float32)
    _bound[(h,)](stats.contiguous(), c, p, triton.next_power_of_2(p), num_warps=8)
    return c


@bound.register_fake
def _(stats):
    return stats.new_empty(stats.shape[:2], dtype=torch.float32)
