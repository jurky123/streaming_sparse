import triton
import triton.language as tl

@triton.jit
def _anchor_logits(q, K, tokens, dims, stride: tl.constexpr, scale: tl.constexpr):
    k = tl.load(K + tokens[None, :] * stride + dims[:, None])
    logits = tl.dot(q, k)
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
        args=[logits, scale * 1.4426950408889634], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _anchor_av(prob, values):
    high = prob.to(values.dtype)
    low = (prob - high.to(tl.float32)).to(values.dtype)
    return tl.dot(high, values) + tl.dot(low, values)


@triton.jit
def _anchored_exact(q, K, V, ids, head, H: tl.constexpr, D: tl.constexpr,
                    count: tl.constexpr, scale: tl.constexpr):
    dims = tl.arange(0, D)
    offsets = tl.arange(0, 64)
    stride: tl.constexpr = H * D
    keys = K + head * D
    values = V + head * D
    tokens = tl.load(ids).to(tl.int32) * 64 + offsets
    scores = _anchor_logits(q, keys, tokens, dims, stride, scale)
    anchor = tl.max(scores, 1)
    prob = tl.exp2(scores - anchor[:, None])
    total = tl.sum(prob, 1)
    accum = _anchor_av(prob, tl.load(values + tokens[:, None] * stride + dims[None, :]))
    for i in tl.range(1, count):
        tokens = tl.load(ids + i).to(tl.int32) * 64 + offsets
        scores = _anchor_logits(q, keys, tokens, dims, stride, scale)
        prob = tl.exp2(scores - anchor[:, None])
        accum += _anchor_av(prob, tl.load(values + tokens[:, None] * stride + dims[None, :]))
        total += tl.sum(prob, 1)
    return accum / total[:, None], anchor + tl.log2(total), total
