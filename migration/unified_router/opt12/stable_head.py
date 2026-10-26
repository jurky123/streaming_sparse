"""Opt11 head surrogate with the same FP16 values but FP32 adjoints throughout."""
import math
import torch
import torch.nn.functional as F
from .precision import fp16_round, HalfBmmFP32Backward


def surrogate(head, d):
    st, score = d['stats'], d['score']
    h, n, _ = score.shape
    center = st[..., 32:64]
    av = center - st[..., :32]
    mean, std = st[..., 66:67], st[..., 67:68]
    ri, aj = d['rem'].long(), d['add'].long()
    c, o = d['costs'], d['origins']

    def side(ix, cv):
        valid = ix >= 0
        diff = torch.where(valid[..., None], cv - center[:, :, None, :], 0.)
        z = torch.where(valid, (score.gather(-1, ix.clamp_min(0)) - mean) / std, 0.)
        raw = torch.stack((diff.square().sum(-1), (diff * av[:, :, None]).sum(-1),
                           z, c[ix.clamp_min(0)] / 64 * valid, valid.float(), o[ix.clamp_min(0)] * valid), -1)
        return torch.cat((raw, torch.zeros((h, n, 1, 6), device=raw.device)), -2), diff

    fi, xi = side(ri, d['cv'][..., :8, :])
    fj, xj = side(aj, d['cv'][..., 8:, :])
    shared = torch.zeros((h, n, 24), device=st.device)
    for k, v in {0: av.square().sum(-1), 6: st[..., 64], 7: st[..., 65], 10: std[..., 0].log(),
                 11: d['beta'], 12: st[..., 68] / st[..., 69], 13: st[..., 70] / 4680,
                 14: math.log1p(d['history_tokens']), 15: math.log1p(4680),
                 16: d['step'] / 3, 23: st[..., 72]}.items():
        shared[..., k] = v
    common = torch.zeros(24, device=st.device, dtype=torch.bool)
    common[[0, 6, 7, 10, 11, 12, 13, 14, 15, 16, 23]] = True
    norm = fp16_round(torch.where(common, (shared - head.mean[:, None]) / head.std[:, None], 0.))
    w = fp16_round(head.w1)
    base = fp16_round((norm[..., None, :] * w[:, None]).sum(-1) + head.b1[:, None])
    ix = torch.tensor([1, 3, 8, 17, 19, 21], device=st.device)
    jx = ix + 1
    fi = fp16_round((fi - head.mean[:, None, None, ix]) / head.std[:, None, None, ix])
    fj = fp16_round((fj - head.mean[:, None, None, jx]) / head.std[:, None, None, jx])
    yi = HalfBmmFP32Backward.apply(fi.reshape(h, n * 9, 6), head.w1[:, :, ix].transpose(1, 2)).reshape(h, n, 9, 16)
    yj = HalfBmmFP32Backward.apply(fj.reshape(h, n * 9, 6), head.w1[:, :, jx].transpose(1, 2)).reshape(h, n, 9, 16)
    cross = torch.matmul(fp16_round(xi), fp16_round(xj).transpose(-1, -2))
    cross = fp16_round((cross - head.mean[:, None, None, None, 5]) / head.std[:, None, None, None, 5])
    zero = fp16_round(-head.mean[:, 5] / head.std[:, 5])[:, None, None].expand(h, n, 17)
    cross = torch.cat((zero, cross.flatten(-2)), -1)
    a = torch.arange(81, device=st.device)
    si = torch.where(a < 17, a - 9, (a - 17) // 8)
    sj = torch.where(a < 9, a - 1, (a - 17) % 8)
    si = torch.where(a >= 9, si, 8)
    sj = torch.where(((a >= 1) & (a <= 8)) | (a >= 17), sj, 8)
    y = base[:, :, None] + yi[:, :, si] + yj[:, :, sj] + cross[..., None] * w[:, None, None, :, 5]
    pred = (F.gelu(y, approximate='tanh') * head.w2[:, None]).sum(-1) + head.b2[:, None]
    return torch.cat((torch.zeros_like(pred[..., :1]), pred[..., 1:]), -1)
