"""Experimental IO-aware content reduction, isolated from the selected adapter.

Each CTA computes multiple pages concurrently and writes moments plus candidate
vectors. No [head, query, page, 32] content tensor is allocated.
"""
import argparse
import math
from pathlib import Path

import torch
import triton as tr
import triton.language as tl

from .common import atomic_json
from .opt2.score import _content
from .opt2.streaming import _reduce_moments
from .opt2.candidate_actions import _candidate_actions


@tr.jit
def _tile_moments(L, LC, V, M, C, R, A, PART, CAND,
                  N: tl.constexpr, B: tl.constexpr, P: tl.constexpr,
                  TOTAL: tl.constexpr, OFFSET: tl.constexpr,
                  GROUPS: tl.constexpr, GOFF: tl.constexpr,
                  GP: tl.constexpr, Q: tl.constexpr, PP: tl.constexpr,
                  DOUBLE: tl.constexpr, RELOAD: tl.constexpr):
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
    # Triton 3.2's batched IEEE dot lowering aborts for this layout. Explicit
    # FP32 FMA keeps pages parallel without invoking that compiler path.
    content = tl.full((GP, Q, 32), 0., tl.float32)
    for j in tl.static_range(P):
        if RELOAD:
            lj = tl.load(L + (row[None, :] * B + page[:, None]) * P + j,
                         valid[:, None] & (qi[None, :] < N), 0.)
            cj = tl.load(LC + (head * B + page) * P + j, valid, float('-inf'))
            wj = tl.exp(lj / math.sqrt(128) + cj[:, None] - maximum) / tl.where(valid[:, None], den, 1.)
            vj = tl.load(V + ((head * B + page[:, None]) * P + j) * 32 + dim[None, :], valid[:, None], 0.)
        else:
            wj = tl.sum(tl.where(pr[None, None, :] == j, weights, 0.), 2)
            vj = tl.sum(tl.where(pr[None, :, None] == j, values, 0.), 1)
        content = tl.fma(wj[:, :, None], vj[:, None, :], content)
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


def score_only(q, basis, projected):
    h, n, _ = q.shape
    total = sum(lc.shape[1] for _, lc, _ in projected)
    score = torch.empty((h, n, total), device=q.device, dtype=torch.float32)
    qq = torch.bmm(q.float(), basis)
    raw, offset = [], 0
    for key, lc, _ in projected:
        b, p = lc.shape[1:]
        logits = torch.bmm(qq, key.transpose(1, 2))
        raw.append(logits)
        _content[(tr.cdiv(n, 32), b, h)](logits, lc, None, score, None,
            n, b, p, total, offset, 32, max(16, p), False, num_warps=4, enable_fp_fusion=True)
        offset += b
    return score, raw


def tiled_stats(q, sketch, score, mask, rem, add, raw, projected, gp=8, tile=16, double=True, warps=4, reload=False):
    h, n, b = score.shape
    groups = sum(tr.cdiv(lc.shape[1], gp) for _, lc, _ in projected)
    partial = torch.empty((h*n, groups, 68), device=q.device, dtype=torch.float64 if double else torch.float32)
    candidates = torch.empty((h*n, 16, 32), device=q.device, dtype=torch.float32)
    stats = torch.empty((h*n, 80), device=q.device, dtype=torch.float32)
    offset = goff = 0
    for logits, (_, lc, value) in zip(raw, projected):
        nb, p = lc.shape[1:]
        _tile_moments[(tr.cdiv(n, tile), tr.cdiv(nb, gp), h)](logits, lc, value,
            mask, sketch.costs, rem, add, partial, candidates, n, nb, p, b,
            offset, groups, goff, gp, tile, max(16, p), double, reload,
            num_warps=warps, num_stages=1, enable_fp_fusion=True)
        offset += nb
        goff += tr.cdiv(nb, gp)
    _reduce_moments[(h*n,)](partial, score, mask, sketch.costs, sketch.origins, stats,
        n, b, tr.next_power_of_2(b), groups, tr.next_power_of_2(groups), num_warps=4, enable_fp_fusion=False)
    return stats, candidates


def finish(q, sketch, score, mask, rem, add, beta, step, raw, projected, head, **settings):
    stats, cv = tiled_stats(q, sketch, score, mask, rem, add, raw, projected, **settings)
    h, n, b = score.shape
    out = torch.empty_like(mask)
    _candidate_actions[(h*n,)](stats, score, cv, mask, sketch.costs, sketch.origins, rem, add,
        head.w1, head.b1, head.w2, head.b2, head.mean, head.std, step, out,
        n, b, tr.next_power_of_2(b), sketch.history_tokens, sketch.current_tokens, beta,
        num_warps=4, enable_fp_fusion=False)
    return out


@torch.no_grad()
def main():
    from .fast_route import FastRoute
    from .opt2_guarded import guarded_finish
    from .opt2_precision import fma_score
    from .opt2.features import _statistics
    from .learning import ActionHead
    from .fast_audit import timed
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    root = Path(args.directory)
    out = root / 'optimization3'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = torch.load(root / 'optimization/stage3_fused_K32/actual_block.pt', map_location='cuda', weights_only=False)
    q = data['q'][:, :512]
    sketch, prepared, beta = data['sketch'], data['prepared'], data['beta']
    head = ActionHead(data['head']).cuda()
    head.load_state_dict(data['head']); head.eval()
    fast = FastRoute(True, True)
    score, content = fma_score(q, *prepared)
    mask = fast.mask(score, beta)
    rem, add = fast.candidates(score, mask, sketch.origins, beta)
    step = q.new_tensor(3., dtype=torch.float32)
    ref = guarded_finish(q, sketch, score, mask, rem, add, beta, step, content, head)
    scores, raw = score_only(q, *prepared)
    assert torch.equal(scores, score)
    refstats = torch.empty((q.shape[0]*q.shape[1], 80), device=q.device, dtype=torch.float32)
    _statistics[(len(refstats),)](score, content, mask, sketch.costs, sketch.origins, refstats,
        q.shape[1], score.shape[-1], tr.next_power_of_2(score.shape[-1]), num_warps=4, enable_fp_fusion=False)
    result = dict(scope='L24 H14040 step3 K32, first512 queries; microbenchmark only',
        score_exact=True, score_only_ms=timed(lambda: score_only(q, *prepared), 30),
        reference_score_ms=timed(lambda: fma_score(q, *prepared), 30),
        reference_finish_ms=timed(lambda: guarded_finish(q, sketch, score, mask, rem, add, beta, step, content, head), 30), settings=[])
    print({k:v for k,v in result.items() if k!='settings'}, flush=True)
    configs = [(8,8,4),(16,8,4),(32,8,4),(16,16,4)] if args.reload else [(4,16,4),(8,16,4),(16,16,4),(8,32,4),(16,16,8)]
    for double in ([False] if args.reload else [False, True]):
        for gp, tile, warps in configs:
            settings = dict(gp=gp, tile=tile, double=double, warps=warps, reload=args.reload)
            actual = finish(q, sketch, score, mask, rem, add, beta, step, raw, prepared[1], head, **settings)
            stats, cv = tiled_stats(q, sketch, score, mask, rem, add, raw, prepared[1], **settings)
            indices = torch.cat((rem, add), -1).reshape(q.shape[0], q.shape[1], 16)
            valid = indices >= 0
            reference_cv = content.gather(2, indices.clamp_min(0)[...,None].expand(-1,-1,-1,32))
            candidate_error = (cv.reshape_as(reference_cv)-reference_cv).abs()[valid]
            groups = sum(tr.cdiv(lc.shape[1],gp) for _,lc,_ in prepared[1])
            row = dict(**settings, finish_ms=timed(lambda: finish(q, sketch, score, mask, rem, add, beta, step, raw, prepared[1], head, **settings),30),
                changed_queries=int((actual!=ref).any(-1).sum()),
                center_max=float((stats[:,:64]-refstats[:,:64]).abs().max()),
                variance_max=float((stats[:,64:66]-refstats[:,64:66]).abs().max()),
                candidates_max=float(candidate_error.max()),
                partial_and_candidate_bytes=q.shape[0]*q.shape[1]*(groups*68*(8 if double else 4)+16*32*4),
                reference_content_bytes=content.numel()*content.element_size())
            result['settings'].append(row)
            print(row, flush=True)
            atomic_json(out/('reload_probe.json' if args.reload else 'tiled_probe.json'), result)


if __name__ == '__main__':
    main()
