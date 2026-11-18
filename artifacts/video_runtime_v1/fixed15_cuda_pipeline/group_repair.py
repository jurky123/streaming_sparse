import triton
import triton as tr
import triton.language as tl
import math
from local_experiments.review_v1.tuned_attention import _av,_anchored_exact
@triton.jit
def selected_online(
    TILE, Q, K, V,
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
    PAGE_K_TILE: tl.constexpr = 0,
    PAGE_V: tl.constexpr = False,
    RETURN_VALUES: tl.constexpr = False,
):
    tl.static_assert(BLOCK_M == 64)
    tl.static_assert(BLOCK_N == 64)
    tl.static_assert(LQ % 64 == 0)
    tl.static_assert(LK * H * D < 2147483648)
    tl.static_assert(LQ * H * D < 2147483648)
    idx_m = TILE.to(tl.int32)
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

    Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

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
        V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * kv_stride + offs_d[None, :]

        if PAGE_K_TILE:
            base = (idx_n * H + idx_h) * BLOCK_N * D
            K_ptrs = K + base + (offs_d[:, None] // PAGE_K_TILE * BLOCK_N + offs_n[None, :]) * PAGE_K_TILE + offs_d[:, None] % PAGE_K_TILE
        if PAGE_V:
            V_ptrs = V + (idx_n * H + idx_h) * BLOCK_N * D + offs_n[:, None] * D + offs_d[None, :]

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
        if RETURN_VALUES:
            # Match the standalone exact kernel inside the non-contracting tail.
            tl.static_assert(HIGH_PRECISION_AV)
            ph = p.to(v.dtype)
            pl = (p - ph.to(tl.float32)).to(v.dtype)
            page_av = tl.dot(ph,v) + tl.dot(pl,v)
            o_s = tl.fma(o_s,alpha[:,None],page_av)
            l_i = tl.fma(l_i,alpha,l_ij)
        else:
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
    if RETURN_VALUES:
        return o_s, m_i + tl.math.log2(l_i)
    else:
        tl.store(OS_ptrs, o_s.to(OS.type.element_ty))

        m_i += tl.math.log2(l_i)
        tl.store(LSE_ptrs, m_i)

@tr.jit
def selected_tail(TILE,Q,KEY,MEAN,DETAIL,MASK,MU,CODE,LSE,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=64,BR:tl.constexpr=64, BF16_AV:tl.constexpr=False,MERGE:tl.constexpr=False,EXACT=None,ELSE=None,BASIS=None,OUT=None,HEAD_PAGES:tl.constexpr=0,K=None,V=None,IDS=None,EXACT_COUNT:tl.constexpr=0,QK_SCALE:tl.constexpr=0.,ANCHOR_SCALE:tl.constexpr=False,COLUMN_MASS:tl.constexpr=False):
 hp:tl.constexpr=HEAD_PAGES if HEAD_PAGES else P
 t=TILE;h=tl.program_id(1);qi=t*BQ+tl.arange(0,BQ);d=tl.arange(0,D);r=tl.arange(0,BR);pages=tl.arange(0,BR//4);rank=tl.arange(0,16)
 q=tl.load(Q+(qi[:,None]*H+h)*D+d[None,:])
 if EXACT_COUNT:
  tl.static_assert((BQ == 64 and BR == 64) and MERGE)
  if ANCHOR_SCALE:
   exact,le,mass=_anchored_exact(q,K,V,IDS+(h*(N//64)+t)*EXACT_COUNT,h,H,D,EXACT_COUNT,QK_SCALE,COLUMN_MASS)
   # The first page contributes at least one. Recompute unsafe tiles using online softmax.
   if tl.min(((mass>=1.)&(mass<1.21e24)).to(tl.int32),0)==0:
    exact,le=selected_online(TILE,Q,K,V,QK_SCALE,EXACT_COUNT,IDS,OUT,OUT,H,N,P*64,N//64,D,64,64,HIGH_PRECISION_AV=True,RETURN_VALUES=True)
  else:
   exact,le=selected_online(TILE,Q,K,V,QK_SCALE,EXACT_COUNT,IDS,OUT,OUT,H,N,P*64,N//64,D,64,64,HIGH_PRECISION_AV=True,RETURN_VALUES=True)
 mx=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32)
 am=tl.zeros((BQ,D),tl.float32);ac=tl.zeros((BQ,16),tl.float32)
 if EXACT_COUNT:
  slots=tl.arange(0,tr.next_power_of_2(EXACT_COUNT))
  ids=tl.load(IDS+(h*(N//64)+t)*EXACT_COUNT+slots,slots<EXACT_COUNT,0)
  bit=tl.where(slots<EXACT_COUNT,1<<(ids%32),0).to(tl.uint32)
  words=tl.sum(tl.where(tl.arange(0,16)[:,None]==ids[None,:]//32,bit[None,:],0),1)
 for first in range(tr.cdiv(P*4,BR)):
  rep=first*BR+r;page=first*(BR//4)+pages
  if EXACT_COUNT:
   word=tl.sum(tl.where(tl.arange(0,16)==first//2,words,0),0)
  else:
   word=tl.load(MASK+(h*(N//64)+(t*BQ)//64)*16+rep//4//32)
  omitted=((word>>(rep//4%32))&1)==0
  key=tl.load(KEY+(h*hp*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
  z=tl.dot(q,key)*(1.4426950408889634/math.sqrt(D))
  z=tl.where((rep[None,:]<P*4)&omitted[None,:],z,-float('inf'))
  nm=tl.maximum(mx,tl.max(z,1));fade=tl.exp2(mx-nm);w=tl.exp2(z-nm[:,None]);ww=tl.reshape(w,(BQ,BR//4,4));rr=tl.arange(0,4)
  alpha=tl.sum(ww,2)
  c0=tl.sum(ww*tl.where(rr<2,1.,-1.)[None,None,:],2)
  c1=tl.sum(ww*tl.where(rr%2==0,1.,-1.)[None,None,:],2)
  c2=tl.sum(ww*tl.where((rr==0)|(rr==3),1.,-1.)[None,None,:],2)
  mean=tl.load(MEAN+(h*hp+page[:,None])*D+d[None,:],page[:,None]<P,0).to(tl.float32)
  x=tl.load(DETAIL+((h*hp+page[:,None])*3)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
  y=tl.load(DETAIL+((h*hp+page[:,None])*3+1)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
  v=tl.load(DETAIL+((h*hp+page[:,None])*3+2)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
  if BF16_AV:
   am=am*fade[:,None]+_av(alpha,mean)
   ac=ac*fade[:,None]+_av(c0,x)+_av(c1,y)+_av(c2,v)
  else:
   am=am*fade[:,None]+tl.dot(alpha,mean,input_precision='tf32x3')
   ac=ac*fade[:,None]+tl.dot(c0,x,input_precision='tf32x3')+tl.dot(c1,y,input_precision='tf32x3')+tl.dot(c2,v,input_precision='tf32x3')
  den=den*fade+tl.sum(w,1);mx=nm
 safe=tl.maximum(den,1.e-30)
 if MERGE:
  code=tl.where(rank[None,:]<8,ac/safe[:,None],0.)
  basis=tl.load(BASIS+(h*D+d[None,:])*8+rank[:,None],rank[:,None]<8,0).to(tl.float32)
  tail=am/safe[:,None]+tl.dot(code,basis,input_precision='tf32x3')
  if not EXACT_COUNT:
   le=tl.load(ELSE+h*N+qi)
   exact=tl.load(EXACT+(qi[:,None]*H+h)*D+d[None,:])
  lt=tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf'))
  rho=1./(1.+tl.exp2(le-lt))
  tl.store(OUT+(qi[:,None]*H+h)*D+d[None,:],exact*(1.-rho[:,None])+tail*rho[:,None])
 else:
  tl.store(MU+(h*N+qi[:,None])*D+d[None,:],am/safe[:,None])
  tl.store(CODE+(h*N+qi[:,None])*8+rank[None,:],ac/safe[:,None],rank[None,:]<8)
  tl.store(LSE+h*N+qi,tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf')))

@tr.jit
def repair_group(FLAGS,Q,K,V,IDS,KEY,MEAN,DETAIL,BASIS,OUT,P:tl.constexpr,HP:tl.constexpr,C:tl.constexpr,G:tl.constexpr):
 h=tl.program_id(1)
 start=tl.program_id(0)*G
 slots=start+tl.arange(0,G)
 bad=tl.load(FLAGS+h*72+slots,slots<72,0)
 if tl.max(bad,0):
  for i in range(G):
   tile=start+i
   if tl.load(FLAGS+h*72+tile,tile<72,0):
    selected_tail(tile,Q,KEY,MEAN,DETAIL,None,None,None,None,P,4608,12,128,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=BASIS,OUT=OUT,HEAD_PAGES=HP,K=K,V=V,IDS=IDS,ANCHOR_SCALE=False,COLUMN_MASS=True,EXACT_COUNT=C,QK_SCALE=128**-.5)
