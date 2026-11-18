"""FP32 pointwise grouping operations around the original PyTorch reductions."""
import torch
import triton
import triton.language as tl


@triton.jit
def _response(A, V, MEAN, R, SQ, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    column = (i // (N * 32)) * 32 + i % 32
    r = tl.load(A + i) * N * (tl.load(V + i) - tl.load(MEAN + column))
    tl.store(R + i, r)
    tl.store(SQ + i, r * r)


@triton.jit
def _finish(S, R, KS, RS, G, OUT, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    h = i // (N * 32)
    k = tl.div_rn(tl.load(S + i), tl.load(KS + h))
    r = tl.div_rn(tl.load(R + i), tl.load(RS + h))
    tl.store(OUT + i, k + tl.load(G + h * 32 + i % 32) * r)


def finish(score, rms, a, vp, mean, gate):
    assert score.is_contiguous() and vp.is_contiguous() and a.is_contiguous()
    n = score.size(-2)
    assert score.shape[1:] == (1, 4608, 32)
    response = torch.empty_like(score)
    squares = torch.empty_like(score)
    grid = (score.numel() // 256,)
    _response[grid](a, vp, mean, response, squares, n, 256, enable_fp_fusion=False)
    response_rms = squares.mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
    out = torch.empty_like(score)
    _finish[grid](score, response, rms, response_rms, gate, out, n, 256, enable_fp_fusion=False)
    return out
