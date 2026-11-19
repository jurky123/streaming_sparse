import torch


def initial(risk, own, quota):
    p = risk.shape[-1]
    pages = torch.arange(p, device=risk.device)
    protected = (pages < 24) | (pages == own[..., None])
    current, history = quota
    ci = risk[..., -72:].masked_fill(protected[..., -72:], -torch.inf).topk(current - 1).indices + p - 72
    hi = risk[..., 24:-72].topk(history).indices + 24
    return protected.scatter(-1, ci, True).scatter(-1, hi, True)


def pool(risk, mask, own):
    p = risk.shape[-1]
    page = torch.arange(p, device=risk.device)
    movable = mask & (page >= 24) & (page != own[..., None])
    removes, adds = [], []
    for lo, hi in ((24, p - 72), (p - 72, p)):
        score = risk[..., lo:hi]
        removes.append(score.masked_fill(~movable[..., lo:hi], torch.inf).argsort(dim=-1, stable=True)[..., :8] + lo)
        adds.append(score.masked_fill(mask[..., lo:hi], -torch.inf).argsort(dim=-1, descending=True, stable=True)[..., :16] + lo)
    r, a = torch.cat(removes, -1), torch.cat(adds, -1)
    legal = movable.gather(-1, r)[..., :, None] & ~mask.gather(-1, a)[..., None, :]
    legal = legal & ((r[..., :, None] >= p - 72) == (a[..., None, :] >= p - 72))
    return r, a, legal


def pairs(risk, mask, own):
    r, a, legal = pool(risk, mask, own)
    i, j = legal.nonzero(as_tuple=True)
    return r[i], a[j]


def state(c, d, s, m, mask, r, a, mode, risk, rng):
    from local_experiments.full_coverage.stable_targets import true_gains
    for _ in range(2 if mode == 7 else int(mode >= 6)):
        if mode >= 8:
            j = int(rng.integers(len(r)))
        else:
            gains = true_gains(c, d, s, m, mask, r, a)
            j = gains.argmax()
            if gains[j] <= 0:
                break
        mask = mask.clone()
        mask[r[j]], mask[a[j]] = False, True
        r, a = pairs(risk, mask, torch.as_tensor(d['own'], device=r.device))
    return mask, r, a
