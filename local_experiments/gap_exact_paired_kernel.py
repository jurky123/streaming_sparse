"""Compute multiple selected physical64 pages per iteration; query tile remains64."""
import triton
import triton.language as tl
@triton.jit
def _attn_fwd(
    Q, K, V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, LSE, OS,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    # Q/K/V/O: (B, L, H, D)  ->  base = b*L*H*D + h*D,  stride_l = H*D
    q_offset = idx_b * LQ * HD + idx_h * D
    kv_offset = idx_b * LK * HD + idx_h * D
    # LUT: (B, H, M_BLOCKS, topk)  ->  flat (B*H, M_BLOCKS, topk)
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    # LSE: (B, H, LQ)  ->  flat (B*H, LQ)
    lse_offset = idx_bh * LQ

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ)
    GROUP: tl.constexpr = BLOCK_N // 64
    for block_idx in tl.range(tl.cdiv(topk, GROUP)):
        lut_index = block_idx * GROUP + offs_n // 64
        idx_n = tl.load(LUT_ptr + lut_index, lut_index < topk, 0).to(tl.int64)
        k_start = idx_n * 64
        k_mask = (lut_index < topk) & ((k_start + offs_n % 64) < LK)

        K_ptrs = K + kv_offset + (k_start + offs_n % 64)[None, :] * HD + offs_d[:, None]
        V_ptrs = V + kv_offset + (k_start + offs_n % 64)[:, None] * HD + offs_d[None, :]

        k = tl.load(K_ptrs, mask=k_mask[None, :], other=0.)
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
        qk = tl.where(k_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs, mask=k_mask[:, None], other=0.)
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < LQ)

    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < LQ)
