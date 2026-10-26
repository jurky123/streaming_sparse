"""Algebraically factored FP32 action head; no weight changes or query sharing."""
import math
import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import erf

@tr.jit
def _factor_actions(ST, X, V, S, C, O, R, A, W1, B1, W2, B2, MU, SIG, STEP, OUT, N: tl.constexpr, B: tl.constexpr, P: tl.constexpr, HIST: tl.constexpr, CUR: tl.constexpr, BETA: tl.constexpr):
    row = tl.program_id(0)
    head = row // N
    p = tl.arange(0, P)
    d = tl.arange(0, 32)
    a = tl.arange(0, 128)
    hidden = tl.arange(0, 16)
    selected = tl.load(S + row * B + p, p < B, False)
    total = tl.load(ST + row * 80 + d)
    center = tl.load(ST + row * 80 + 32 + d)
    av = center - total
    ds = tl.load(ST + row * 80 + 64)
    da = tl.load(ST + row * 80 + 65)
    mean = tl.load(ST + row * 80 + 66)
    std = tl.load(ST + row * 80 + 67)
    scost = tl.load(ST + row * 80 + 68)
    cost = tl.load(ST + row * 80 + 69)
    current_cost = tl.load(ST + row * 80 + 70)
    nselected = tl.load(ST + row * 80 + 71).to(tl.int32)
    meanz = tl.load(ST + row * 80 + 72)
    remove_slot = tl.where(a < 17, a - 9, (a - 17) // 8)
    add_slot = tl.where(a < 9, a - 1, (a - 17) % 8)
    i = tl.load(R + row * 8 + remove_slot, (a >= 9) & (a < 81), -1)
    j = tl.load(A + row * 8 + add_slot, (a >= 1) & (a <= 8) | (a >= 17) & (a < 81), -1)
    vi = tl.load(V + (row * B + i[:, None]) * 32 + d[None, :], i[:, None] >= 0, 0.0)
    vj = tl.load(V + (row * B + j[:, None]) * 32 + d[None, :], j[:, None] >= 0, 0.0)
    xi = tl.where(i[:, None] >= 0, vi - center[None, :], 0.0)
    xj = tl.where(j[:, None] >= 0, vj - center[None, :], 0.0)
    g0 = tl.sum(av * av, 0)
    g1 = tl.sum(xi * xi, 1)
    g2 = tl.sum(xj * xj, 1)
    g3 = tl.sum(av[None, :] * xi, 1)
    g4 = tl.sum(av[None, :] * xj, 1)
    g5 = tl.sum(xi * xj, 1)
    zi = tl.where(i >= 0, (tl.load(X + row * B + i, i >= 0, 0.0) - mean) / std, 0.0)
    zj = tl.where(j >= 0, (tl.load(X + row * B + j, j >= 0, 0.0) - mean) / std, 0.0)
    ci = tl.load(C + i, i >= 0, 0.0)
    cj = tl.load(C + j, j >= 0, 0.0)
    oi = tl.load(O + i, i >= 0, 0.0)
    oj = tl.load(O + j, j >= 0, 0.0)
    
    step = tl.load(STEP)
    # Separate the action-independent head contribution from the 13 variable
    # features. This is the same affine layer, evaluated with fewer repeated ops.
    shared = tl.full((32,), 0., tl.float32)
    shared = tl.where(d==0,g0,shared)
    shared = tl.where(d==6,ds,shared);shared=tl.where(d==7,da,shared)
    shared = tl.where(d==10,tl.log(std),shared);shared=tl.where(d==11,BETA,shared)
    shared = tl.where(d==12,scost/cost,shared);shared=tl.where(d==13,current_cost/CUR,shared)
    shared = tl.where(d==14,math.log1p(HIST),shared);shared=tl.where(d==15,math.log1p(CUR),shared)
    shared = tl.where(d==16,step/3.,shared);shared=tl.where(d==23,meanz,shared)
    mu=tl.load(MU+head*24+d,d<24,0.);sig=tl.load(SIG+head*24+d,d<24,1.)
    w=tl.load(W1+head*384+hidden[None,:]*24+d[:,None],d[:,None]<24,0.)
    base=tl.sum(((shared-mu)/sig)[:,None]*w,0)+tl.load(B1+head*16+hidden)
    y=tl.broadcast_to(base[None,:],(128,16))
    weight=tl.load(W1+head*384+hidden*24+1)/tl.load(SIG+head*24+1)
    y=y+(g1)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+2)/tl.load(SIG+head*24+2)
    y=y+(g2)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+3)/tl.load(SIG+head*24+3)
    y=y+(g3)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+4)/tl.load(SIG+head*24+4)
    y=y+(g4)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+5)/tl.load(SIG+head*24+5)
    y=y+(g5)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+8)/tl.load(SIG+head*24+8)
    y=y+(zi)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+9)/tl.load(SIG+head*24+9)
    y=y+(zj)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+17)/tl.load(SIG+head*24+17)
    y=y+(ci/64.)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+18)/tl.load(SIG+head*24+18)
    y=y+(cj/64.)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+19)/tl.load(SIG+head*24+19)
    y=y+((i>=0).to(tl.float32))[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+20)/tl.load(SIG+head*24+20)
    y=y+((j>=0).to(tl.float32))[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+21)/tl.load(SIG+head*24+21)
    y=y+(oi)[:,None]*weight[None,:]
    weight=tl.load(W1+head*384+hidden*24+22)/tl.load(SIG+head*24+22)
    y=y+(oj)[:,None]*weight[None,:]
    y = 0.5 * y * (1.0 + erf(y * 0.7071067811865476))
    w2 = tl.load(W2 + head * 16 + hidden)
    b2 = tl.load(B2 + head)
    pred = tl.sum(y * w2[None, :], 1) + b2
    valid = (a == 0) | (a >= 1) & (a <= 8) & (j >= 0) | (a >= 9) & (a <= 16) & (i >= 0) & (nselected > 1) | (a >= 17) & (a < 81) & (i >= 0) & (j >= 0)
    pred = tl.where(a == 0, 0.0, tl.where(valid, pred, float('-inf')))
    best = tl.max(pred, 0)
    chosen = tl.min(tl.where(pred == best, a, 2147483647), 0)
    ri = tl.sum(tl.where(a == chosen, i, 0), 0)
    aj = tl.sum(tl.where(a == chosen, j, 0), 0)
    final = selected & (p != ri) | (p == aj)
    tl.store(OUT + row * B + p, final, p < B)
