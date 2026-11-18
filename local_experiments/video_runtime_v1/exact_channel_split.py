"""Measure partitioning the output channels across CTAs while preserving each AV chain."""
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
    HIGH_PRECISION_AV: tl.constexpr = False,
    HEAD_MAJOR_KV: tl.constexpr = False,
    CACHE_LUT: tl.constexpr = False,
    PREFETCH_PAGES: tl.constexpr = 0,
    QUERY_GROUP: tl.constexpr = 0,
    KV_HEAD_STRIDE: tl.constexpr = 0,
    VD: tl.constexpr = 64,
):
    tl.static_assert(BLOCK_M == 64)
    tl.static_assert(BLOCK_N == 64)
    tl.static_assert(LQ % 64 == 0)
    tl.static_assert(LK * H * D < 2147483648)
    tl.static_assert(LQ * H * D < 2147483648)
    idx_m = tl.program_id(0).to(tl.int32)
    idx_bh = tl.program_id(1).to(tl.int32)

    if QUERY_GROUP:
        tl.static_assert(M_BLOCKS % QUERY_GROUP == 0)
        linear = idx_bh * M_BLOCKS + idx_m
        idx_m = linear // (H * QUERY_GROUP) * QUERY_GROUP + linear % QUERY_GROUP
        idx_bh = linear // QUERY_GROUP % H
    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    # Q/K/V/O: (B, L, H, D)  ->  base = b*L*H*D + h*D,  stride_l = H*D
    q_offset = idx_b * LQ * HD + idx_h * D
    kv_offset = idx_b * LK * HD + idx_h * D
    # LUT: (B, H, M_BLOCKS, topk)  ->  flat (B*H, M_BLOCKS, topk)
    kv_stride: tl.constexpr = D if HEAD_MAJOR_KV else HD
    if HEAD_MAJOR_KV:
        kv_offset = idx_bh * (KV_HEAD_STRIDE if KV_HEAD_STRIDE else LK * D)
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    # LSE: (B, H, LQ)  ->  flat (B*H, LQ)
    lse_offset = idx_bh * LQ

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    offs_v = tl.program_id(2)*VD + tl.arange(0,VD)
    Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * HD + offs_v[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, VD], dtype=tl.float32)

    q = tl.load(Q_ptrs)
    if CACHE_LUT:
        positions = tl.arange(0, triton.next_power_of_2(topk))
        page_ids = tl.load(LUT_ptr + positions, positions < topk, 0).to(tl.int32)
    for block_idx in tl.range(topk):
        if CACHE_LUT:
            idx_n = tl.sum(tl.where(positions == block_idx, page_ids, 0), 0)
        else:
            idx_n = tl.load(LUT_ptr + block_idx).to(tl.int32)
        if PREFETCH_PAGES:
            if block_idx + PREFETCH_PAGES < topk:
                future = tl.load(LUT_ptr + block_idx + PREFETCH_PAGES).to(tl.int32)
                line = tl.arange(0,128)
                offset = kv_offset + (future*BLOCK_N + line//2)*kv_stride + line%2*64
                tl.inline_asm_elementwise("prefetch.global.L2 [$1]; prefetch.global.L2 [$2]; mov.u32 $0, 0;",
                                          constraints="=r,l,l",args=[K+offset,V+offset],
                                          dtype=tl.int32,is_pure=False,pack=1)
        k_start = idx_n * BLOCK_N
        k_mask = (k_start + offs_n) < LK

        K_ptrs = K + kv_offset + (k_start + offs_n)[None, :] * kv_stride + offs_d[:, None]
        V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * kv_stride + offs_v[None, :]

        k = tl.load(K_ptrs)
        qk = tl.dot(q, k)
        qk = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[qk, qk_scale * 1.4426950408889634], dtype=tl.float32, is_pure=True, pack=1)

        v = tl.load(V_ptrs)
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        if HIGH_PRECISION_AV:
            ph=p.to(v.dtype)
            pl=(p-ph.to(tl.float32)).to(v.dtype)
            o_s += tl.dot(ph,v)+tl.dot(pl,v)
        else:
            o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty))

    m_i += tl.math.log2(l_i)
    if tl.program_id(2)==0:
        tl.store(LSE_ptrs, m_i)
