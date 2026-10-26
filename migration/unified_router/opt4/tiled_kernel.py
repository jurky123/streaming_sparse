"""Page-tiled moments using Tensor Cores, with FP32 accumulators."""
import math
import triton as tr
import triton.language as tl

@tr.jit
def _tile_moments(L, LC, V, M, C, R, A, PART, CAND,
                  N: tl.constexpr, B: tl.constexpr, P: tl.constexpr,
                  TOTAL: tl.constexpr, OFFSET: tl.constexpr,
                  GROUPS: tl.constexpr, GOFF: tl.constexpr,
                  GP: tl.constexpr, Q: tl.constexpr, PP: tl.constexpr,
                  DOUBLE: tl.constexpr, PRECISION: tl.constexpr):
    qi = tl.program_id(0) * Q + tl.arange(0, Q)
    pg = tl.program_id(1)
    head = tl.program_id(2)
    page = pg * GP + tl.arange(0, GP)
    pr = tl.arange(0, PP)
    dim = tl.arange(0, 32)
    slot = tl.arange(0, 8)
    valid = page < B
    row = head * N + qi
    ll = tl.load(L + ((row[None, :, None] * B + page[:, None, None]) * P + pr[None, None, :]),
                 valid[:, None, None] & (qi[None, :, None] < N) & (pr[None, None, :] < P), 0.)
    lc = tl.load(LC + (head * B + page[:, None]) * P + pr[None, :],
                 valid[:, None] & (pr[None, :] < P), float('-inf'))
    ll = ll / math.sqrt(128) + lc[:, None, :]
    maximum = tl.max(ll, 2)
    maximum = tl.where(valid[:, None], maximum, 0.)
    weights = tl.exp(ll - maximum[:, :, None])
    den = tl.sum(weights, 2)
    weights = weights / tl.where(valid[:, None], den, 1.)[:, :, None]
    values = tl.load(V + ((head * B + page[:, None, None]) * P + pr[None, :, None]) * 32 + dim[None, None, :],
                     valid[:, None, None] & (pr[None, :, None] < P), 0.)
    if PRECISION == 'fp16':
        content = tl.dot(weights.to(tl.float16), values.to(tl.float16))
    else:
        content = tl.dot(weights, values, input_precision=PRECISION)
    cost = tl.load(C + OFFSET + page, valid, 0.)
    selected = tl.load(M + row[None, :] * TOTAL + OFFSET + page[:, None],
                       valid[:, None] & (qi[None, :] < N), False)
    sc = cost[:, None] * selected
    dtype: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    v = content.to(dtype)
    totalnum = tl.sum(v * cost[:, None, None], 0)
    selnum = tl.sum(v * sc[:, :, None], 0)
    square = tl.sum(v * v, 2)
    e2 = tl.sum(square * cost[:, None], 0)
    se2 = tl.sum(square * sc, 0)
    cost_sum = tl.sum(cost, 0)
    scost_sum = tl.sum(sc, 0)
    base = (row[:, None] * GROUPS + GOFF + pg) * 68
    tl.store(PART + base + dim[None, :], totalnum, qi[:, None] < N)
    tl.store(PART + base + 32 + dim[None, :], selnum, qi[:, None] < N)
    scalar = (row * GROUPS + GOFF + pg) * 68
    tl.store(PART + scalar + 64, e2, qi < N)
    tl.store(PART + scalar + 65, se2, qi < N)
    tl.store(PART + scalar + 66, cost_sum, qi < N)
    tl.store(PART + scalar + 67, scost_sum, qi < N)
    remove = tl.load(R + row[:, None] * 8 + slot[None, :], qi[:, None] < N, -1)
    add = tl.load(A + row[:, None] * 8 + slot[None, :], qi[:, None] < N, -1)
    ri = tl.min(tl.where(remove[None, :, :] == (OFFSET + page[:, None, None]), slot[None, None, :], 8), 2)
    aj = tl.min(tl.where(add[None, :, :] == (OFFSET + page[:, None, None]), slot[None, None, :], 8), 2)
    tl.store(CAND + (row[None, :, None] * 16 + ri[:, :, None]) * 32 + dim[None, None, :], content,
             valid[:, None, None] & (qi[None, :, None] < N) & (ri[:, :, None] < 8))
    tl.store(CAND + (row[None, :, None] * 16 + 8 + aj[:, :, None]) * 32 + dim[None, None, :], content,
             valid[:, None, None] & (qi[None, :, None] < N) & (aj[:, :, None] < 8))

