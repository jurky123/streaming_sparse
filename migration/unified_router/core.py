"""Online prototype routing and a separate exact offline LabelStore.

No online signature accepts true page LogZ, Dense outputs or current Attention.
History is distance-P8/P4/P2; current is P16. First generation stays Dense.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass
class PagePart:
    key: torch.Tensor
    value: torch.Tensor
    count: torch.Tensor


@dataclass
class Sketch:
    parts: list
    costs: torch.Tensor
    origins: torch.Tensor
    history_tokens: int
    current_tokens: int
    key_basis: torch.Tensor
    value_basis: torch.Tensor

    @property
    def current(self):
        return self.parts[-1]


def page_costs(n, device):
    result = torch.full((math.ceil(n / 64),), 64., device=device)
    if n % 64:
        result[-1] = n % 64
    return result


def part(k, v, weight, bias):
    h, n, d = k.shape
    p = weight.shape[1]
    b = math.ceil(n / 64)
    if n == 0:
        return PagePart(k.new_empty(h, 0, p, d, dtype=torch.float32),
                        k.new_empty(h, 0, p, d, dtype=torch.float32),
                        k.new_empty(h, 0, p, dtype=torch.float32))
    kk = F.pad(k.float(), (0, 0, 0, b * 64 - n)).reshape(h, b, 64, d)
    vv = F.pad(v.float(), (0, 0, 0, b * 64 - n)).reshape_as(kk) if v is not None else None
    logits = torch.bmm(kk.flatten(1, 2), weight.transpose(1, 2)).reshape(h, b, 64, p)
    valid = torch.arange(b * 64, device=k.device).reshape(b, 64) < n
    a = (logits + bias[:, None, None]).softmax(-1) * valid[None, ..., None]
    count = a.sum(2)
    denom = count.clamp_min(1e-20)[..., None]
    return PagePart(torch.einsum('hbtp,hbtd->hbpd', a, kk) / denom,
                    torch.einsum('hbtp,hbtd->hbpd', a, vv) / denom if vv is not None else None, count)


def build(kh, vh, kc, vc, parameters, need_value=True):
    parts, costs, origins = [], [], []
    for start in range(0, kh.shape[1], 4680):
        end = min(start+4680, kh.shape[1])
        age = math.ceil((kh.shape[1]-start)/4680)
        p = 8 if age == 1 else 4 if age == 2 else 2
        parts.append(part(kh[:, start:end], vh[:, start:end] if need_value else None,
                          parameters[f'history_{p}_weight'], parameters[f'history_{p}_bias']))
        costs.append(page_costs(end-start, kh.device))
        origins.append(torch.zeros(len(costs[-1]), device=kh.device))
    parts.append(part(kc, vc if need_value else None, parameters['current_weight'], parameters['current_bias']))
    assert parts[-1].count.shape[-1] == 16
    costs.append(page_costs(kc.shape[1], kc.device))
    origins.append(torch.ones(len(costs[-1]), device=kc.device))
    return Sketch(parts, torch.cat(costs), torch.cat(origins), kh.shape[1], kc.shape[1],
                  parameters['key_basis'], parameters['value_basis'])


def scores(q, sketch, key_dim=64, content='V32'):
    assert key_dim in (32, 64, 128)
    basis = sketch.key_basis[..., :key_dim] if key_dim != 128 else None
    qq = torch.bmm(q.float(), basis) if basis is not None else q.float()
    out = []
    contents = []
    for p in sketch.parts:
        if p.key.shape[1] == 0:
            continue
        h, b, prototypes, _ = p.key.shape
        kk = torch.bmm(p.key.flatten(1, 2), basis).reshape(h, b, prototypes, key_dim) if basis is not None else p.key
        ll = torch.bmm(qq, kk.flatten(1, 2).transpose(1, 2)).reshape(h, len(q[0]), b, prototypes) / math.sqrt(128)
        ll = ll + p.count.clamp_min(1e-30).log()[:, None]
        ll = ll.masked_fill((p.count <= 1e-8)[:, None], -torch.inf)
        out.append(ll.logsumexp(-1))
        if content is not None:
            if content == 'K32':
                values = torch.bmm(p.key.flatten(1, 2), sketch.key_basis[..., :32]).reshape(h, b, prototypes, 32)
            else:
                r = 8 if content == 'V8' else 32
                values = torch.bmm(p.value.flatten(1, 2), sketch.value_basis[..., :r]).reshape(h, b, prototypes, r)
            contents.append(torch.einsum('hqbp,hbpr->hqbr', ll.softmax(-1), values))
    return torch.cat(out, 2), torch.cat(contents, 2) if content is not None else None


def initial_mask(score, sketch, beta, mode='U-Sol'):
    if not torch.isfinite(score).all():
        raise FloatingPointError('Nonfinite proxy score')
    selected = torch.ones_like(score, dtype=torch.bool)
    if mode == 'Dense' or sketch.history_tokens == 0:
        return selected
    editable = torch.ones_like(sketch.origins, dtype=torch.bool)
    if mode == 'H-Sol':
        editable = sketch.origins == 0
    elif mode == 'C-Sol':
        editable = sketch.origins == 1
    elif mode != 'U-Sol':
        raise ValueError(mode)
    values = score[..., editable]
    if values.shape[-1] == 0:
        return selected
    std, mean = torch.std_mean(values, dim=-1, unbiased=False, keepdim=True)
    keep = values > mean + beta * std
    keep = torch.where(std <= 1e-6, torch.ones_like(keep), keep)
    selected[..., editable] = keep
    empty = ~selected.any(-1)
    selected.scatter_(-1, score.argmax(-1, keepdim=True),
        selected.gather(-1, score.argmax(-1, keepdim=True)) | empty[..., None])
    return selected


def candidates(score, mask, origins, beta, seed=0):
    """Each side:3 boundary H +3 boundary C +2 exploration, then fill holes."""
    shape = score.shape
    x = score.flatten(0, 1)
    selected = mask.flatten(0, 1)
    std, mean = torch.std_mean(x, dim=-1, unbiased=False, keepdim=True)
    distance = (x - mean - beta * std).abs()
    n, b = x.shape
    page = torch.arange(b, device=x.device)[None]
    hashes = ((page * 1103515245 + torch.arange(n, device=x.device)[:, None] * 12345 + seed) % 2147483647).float()

    def side(pool):
        remaining = pool.clone()
        parts = []

        def take(rank_value, allowed, count):
            values = rank_value.masked_fill(~allowed, torch.inf)
            order = torch.argsort(values, dim=-1, stable=True)[:, :min(count, b)]
            valid = allowed.gather(-1, order)
            result = order.masked_fill(~valid, -1)
            remaining.scatter_(-1, order, remaining.gather(-1, order) & ~valid)
            if result.shape[-1] < count:
                result = F.pad(result, (0, count-result.shape[-1]), value=-1)
            return result

        for origin in [0, 1]:
            parts.append(take(distance, remaining & (origins[None] == origin), 3))
        parts.append(take(hashes, remaining, 2))
        result = torch.cat(parts, -1)
        for column in range(8):
            missing = result[:, column] < 0
            choice = distance.masked_fill(~remaining, torch.inf).argmin(-1)
            valid = missing & remaining.any(-1)
            result[:, column] = torch.where(valid, choice, result[:, column])
            remaining.scatter_(-1, choice[:, None], remaining.gather(-1, choice[:, None]) & ~valid[:, None])
        return result

    remove, add = side(selected), side(~selected)
    absent = torch.full((n, 1), -1, device=x.device, dtype=torch.long)
    i = torch.cat((absent, absent.expand(-1, 8), remove,
                   remove.repeat_interleave(8, -1)), -1)
    j = torch.cat((absent, add, absent.expand(-1, 8), add.repeat(1, 8)), -1)
    types = torch.tensor([0] + [1]*8 + [2]*8 + [3]*64, device=x.device)
    valid = torch.cat((torch.ones(n, 1, device=x.device, dtype=torch.bool), add >= 0,
        (remove >= 0) & (selected.sum(-1, keepdim=True) > 1),
        (remove.repeat_interleave(8, -1) >= 0) & (add.repeat(1, 8) >= 0)), -1)
    if not (origins == 0).any():
        valid[:, 1:] = False
    return dict(remove=i, add=j, valid=valid, types=types, shape=shape)


def apply_action(mask, actions, chosen):
    flat = mask.flatten(0, 1).clone()
    rows = torch.arange(len(flat), device=flat.device)
    i = actions['remove'][rows, chosen]
    j = actions['add'][rows, chosen]
    flat[rows[i >= 0], i[i >= 0]] = False
    flat[rows[j >= 0], j[j >= 0]] = True
    assert flat.any(-1).all()
    return flat.reshape_as(mask)


def features(q, sketch, score, mask, actions, beta, step, content, no_set=False):
    """24 deployable scalars only. LabelStore is not an accepted argument."""
    x = score.flatten(0, 1)
    v = content.flatten(0, 1)
    s = mask.flatten(0, 1)
    costs = sketch.costs[None].expand(len(x), -1)
    std, mean = torch.std_mean(x, dim=-1, unbiased=False, keepdim=True)
    z = (x-mean) / std.clamp_min(1e-6)
    total = (costs[..., None]*v).sum(1) / costs.sum(1, keepdim=True)
    selected_costs = costs*s
    center = (selected_costs[..., None]*v).sum(1) / selected_costs.sum(1, keepdim=True)
    a = center-total
    ds = ((v-center[:, None]).square().sum(-1)*selected_costs).sum(-1)/selected_costs.sum(-1)
    da = ((v-total[:, None]).square().sum(-1)*costs).sum(-1)/costs.sum(-1)
    if no_set:
        means = []
        variances = []
        for origin in [0, 1]:
            w = costs*(sketch.origins[None] == origin)
            mu = (v*w[..., None]).sum(1)/w.sum(1, keepdim=True).clamp_min(1)
            means.append(mu)
            variances.append(((v-mu[:, None]).square().sum(-1)*w).sum(-1)/w.sum(-1).clamp_min(1))
        center = total
        a = means[0]-means[1]
        ds, da = variances
    i, j = actions['remove'], actions['add']
    pi, pj = (i >= 0).float(), (j >= 0).float()

    def vector(index):
        value = v.gather(1, index.clamp_min(0)[..., None].expand(-1, -1, v.shape[-1]))
        return (value-center[:, None])*(index >= 0)[..., None]

    xi, xj = vector(i), vector(j)
    aa = a[:, None].expand_as(xi)
    geometry = torch.stack((aa.square().sum(-1), xi.square().sum(-1), xj.square().sum(-1),
        (aa*xi).sum(-1), (aa*xj).sum(-1), (xi*xj).sum(-1),
        ds[:, None].expand_as(i), da[:, None].expand_as(i)), -1)
    gather = lambda value, index: value.gather(-1, index.clamp_min(0))*(index >= 0)
    const = lambda value: torch.full_like(i, float(value), dtype=torch.float32)
    cc = sketch.origins[None].expand_as(x)
    scalar = torch.stack((gather(z, i), gather(z, j), std.clamp_min(1e-6).log().expand_as(i),
        const(beta), (selected_costs.sum(-1)/costs.sum(-1))[:, None].expand_as(i),
        ((selected_costs*cc).sum(-1)/sketch.current_tokens)[:, None].expand_as(i),
        const(math.log1p(sketch.history_tokens)), const(math.log1p(sketch.current_tokens)),
        const(step/3), gather(costs, i)/64, gather(costs, j)/64, pi, pj,
        gather(cc, i), gather(cc, j), ((z*s).sum(-1)/s.sum(-1))[:, None].expand_as(i)), -1)
    assert geometry.shape[-1] == 8 and scalar.shape[-1] == 16
    return geometry, scalar


class LabelStore:
    """Exact original128-d page statistics; unavailable to online routing."""
    def __init__(self, q, kh, vh, kc, vc):
        self.history_tokens = kh.shape[1]
        self.current_tokens = kc.shape[1]
        all_logits = torch.bmm(q.float(), torch.cat((kh, kc), 1).float().transpose(1, 2))/math.sqrt(128)
        lz, mu = [], []
        start = 0
        segments = [(kh[:, s:s+4680], vh[:, s:s+4680]) for s in range(0, kh.shape[1], 4680)] + [(kc, vc)]
        for k, v in segments:
            n = k.shape[1]
            if not n:
                continue
            b = math.ceil(n/64)
            lp = F.pad(all_logits[..., start:start+n], (0, b*64-n), value=-torch.inf)
            lp = lp.reshape(len(q), len(q[0]), b, 64).double()
            vv = F.pad(v.double(), (0, 0, 0, b*64-n)).reshape(len(q), b, 64, 128)
            lz.append(lp.logsumexp(-1))
            mu.append(torch.einsum('hqbt,hbtd->hqbd', lp.softmax(-1), vv))
            start += n
        self.logz = torch.cat(lz, 2)
        self.value = torch.cat(mu, 2)
        self.target = self.output(torch.ones_like(self.logz, dtype=torch.bool))

    def output(self, mask):
        assert mask.any(-1).all()
        weights = self.logz.masked_fill(~mask, -torch.inf).softmax(-1)
        return torch.einsum('hqb,hqbd->hqd', weights, self.value)

    def action_labels(self, mask, actions, covariance=False):
        base = self.output(mask)
        error = base-self.target
        base_error = error.square().sum(-1)
        labels = []
        cov = torch.zeros(len(mask), 128, 128, device=mask.device, dtype=torch.float64) if covariance else None
        counts = {t: actions['valid'][:, actions['types'] == t].sum(-1).reshape(mask.shape[:2]) for t in [1, 2, 3]}
        active_types = sum((n > 0).long() for n in counts.values()).clamp_min(1)
        for a in range(81):
            valid = actions['valid'][:, a].reshape(mask.shape[:2])
            chosen = torch.full((mask.shape[0]*mask.shape[1],), a, device=mask.device, dtype=torch.long)
            chosen = torch.where(actions['valid'][:, a], chosen, torch.zeros_like(chosen))
            out = self.output(apply_action(mask, actions, chosen))
            delta = out-base
            gain = base_error-(out-self.target).square().sum(-1)
            labels.append(torch.where(valid, gain, torch.zeros_like(gain)))
            if covariance and a:
                t = int(actions['types'][a])
                weight = valid.double()/counts[t].clamp_min(1)/active_types
                cov += 2*torch.einsum('hq,hqd,hqe->hde', weight, delta, delta)/len(delta[0])
        if covariance:
            cov += torch.einsum('hqd,hqe->hde', error, error)/len(error[0])
        return torch.stack(labels, -1).float(), base_error.float(), cov

    def metrics(self, mask):
        out = self.output(mask)
        return ((out-self.target).square().sum((1, 2))/self.target.square().sum((1, 2)).clamp_min(1e-12)).float()
