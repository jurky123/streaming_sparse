"""Exact deployment forward, explicit differentiable routing/attention surrogate.

The objective is current hard-kernel output NMSE, never cached action labels.
Backward is biased: discrete selection, FP rounding and Skip0 have no useful
ordinary derivative. We use a smooth page gate and a full-attention page-gate
Jacobian. No claim of an exact derivative through the deployment kernel.
"""
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from ..core import build
from ..fast_route import prepare
from ..opt10.compact_route import CompactMomentRoute
from ..opt9.select import fused_select
from ..opt11.head import actions, apply, exact
from .stable_head import surrogate
from .precision import align, HalfBmmFP32Backward, fp16_round
from ..learning import ActionHead
from ..opt11.data import attend


class ActualAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, actual, logz, value):
        mass = (logz - logz.amax(-1, keepdim=True)).exp()
        denom = (mass * gate).sum(-1, keepdim=True).clamp_min(1e-20)
        weights = mass * gate / denom
        smooth = (weights[..., None] * value).sum(-2)
        ctx.save_for_backward(mass / denom, value, smooth)
        return actual.clone()

    @staticmethod
    def backward(ctx, grad):
        scale, value, smooth = ctx.saved_tensors
        gate_grad = scale * ((value - smooth[..., None, :]) * grad[..., None, :]).sum(-1)
        return gate_grad, None, None, None


@torch.no_grad()
def page_statistics(q, raw):
    """Original-128d page logZ and means, ONLY for the backward surrogate.

    They never enter router features or deployment decisions. Recomputed from
    raw QKV; these are not stale counterfactual action labels.
    """
    logzs, values = [], []
    segments = [(raw['kh'][:, s:s+4680], raw['vh'][:, s:s+4680])
                for s in range(0, raw['kh'].shape[1], 4680)]
    segments.append((raw['kc'], raw['vc']))
    for k, v in segments:
        h, n, dim = k.shape
        pages = math.ceil(n / 64)
        logits = torch.bmm(q.float(), k.float().transpose(1, 2)) / math.sqrt(dim)
        logits = F.pad(logits, (0, pages * 64 - n), value=-torch.inf).reshape(h, q.shape[1], pages, 64)
        vp = F.pad(v.float(), (0, 0, 0, pages * 64 - n)).reshape(h, pages, 64, dim)
        logzs.append(logits.logsumexp(-1))
        values.append(torch.einsum('hqbt,hbtd->hqbd', logits.softmax(-1), vp))
    return torch.cat(logzs, 2), torch.cat(values, 2)


class JointRouter(nn.Module):
    def __init__(self, base, parent, layer, variant, beta):
        super().__init__()
        state = torch.load(Path(base) / f'action_basis/layer{layer:02d}.pt', map_location='cpu', weights_only=True)['parameters']
        # Only K64 is used online; preserve inactive columns and value basis.
        self.routing = nn.ParameterDict({k: nn.Parameter(v.clone()) for k, v in state.items()
                                         if k not in ('key_basis', 'value_basis')})
        self.routing['key_projection'] = nn.Parameter(state['key_basis'][..., :64].clone())
        self.register_buffer('key_tail', state['key_basis'][..., 64:].clone())
        self.register_buffer('value_basis', state['value_basis'].clone())
        head = torch.load(Path(parent) / f'strong_training/layer{layer:02d}/{variant}/best.pt', map_location='cpu', weights_only=True)['head']
        self.head = ActionHead(head)
        self.head.load_state_dict(head)
        self.beta = beta
        self.variant = variant
        self.route = CompactMomentRoute(batch=4680)
        self.exact_build = torch.compile(build, fullgraph=True, dynamic=False)
        self.exact_prepare = torch.compile(prepare, fullgraph=True, dynamic=False)

    def dictionary(self):
        p = {k: v for k, v in self.routing.items() if k != 'key_projection'}
        p['key_basis'] = torch.cat((self.routing['key_projection'], self.key_tail), -1)
        p['value_basis'] = self.value_basis
        return p

    @torch.no_grad()
    def deployed_inputs(self, q, raw, step):
        sketch = self.exact_build(raw['kh'], raw['vh'], raw['kc'], raw['vc'], self.dictionary(), need_value=False)
        prepared = self.exact_prepare(sketch, 'K32-Set')
        state = self.route.begin(prepared, sketch)
        score, weights = self.route.score(q, state)
        mask, rem, add = fused_select(score, sketch.origins, self.beta, mode='packed32')
        stats, cv = self.route.stats(q, sketch, score, mask, rem, add, weights, state)
        h, n, _ = score.shape
        stats = stats.reshape(h, n, 80)
        stats[..., 73:] = 0.
        return dict(stats=stats, cv=cv.reshape(h, n, 16, 32), score=score,
                    base_mask=mask, rem=rem.reshape(h, n, 8), add=add.reshape(h, n, 8),
                    costs=sketch.costs, origins=sketch.origins, history_tokens=sketch.history_tokens,
                    step=step, beta=self.beta)

    def differentiable_inputs(self, q, raw, exact_d):
        sketch = build(raw['kh'], raw['vh'], raw['kc'], raw['vc'], self.dictionary(), need_value=False)
        basis = sketch.key_basis[..., :64]
        qq = HalfBmmFP32Backward.apply(q, basis)
        scores, contents, moments = [], [], []
        for part in sketch.parts:
            h, b, p, _ = part.key.shape
            key = torch.bmm(part.key.flatten(1, 2), basis).reshape(h, b, p, 64)
            # Identical half BMM forward, FP32 gradients through the rounded operands.
            logits = HalfBmmFP32Backward.apply(qq, key.flatten(1, 2).transpose(1, 2))
            logits = logits.reshape(h, q.shape[1], b, p) / math.sqrt(128)
            logits = logits + part.count.clamp_min(1e-30).log()[:, None]
            weights = logits.softmax(-1)
            content = fp16_round(key[..., :32])
            scores.append(logits.logsumexp(-1))
            contents.append(torch.einsum('hqbp,hbpd->hqbd', weights, content))
            moments.append(torch.einsum('hqbp,hbp->hqb', weights, content.square().sum(-1)))
        score = align(torch.cat(scores, 2), exact_d['score'])
        content = torch.cat(contents, 2)
        energy = torch.cat(moments, 2)
        mask = exact_d['base_mask'].float()
        cost = exact_d['costs']
        weighted = mask * cost
        den = weighted.sum(-1, keepdim=True)
        total = (content * cost[None, None, :, None]).sum(-2) / cost.sum()
        center = (content * weighted[..., None]).sum(-2) / den
        ds = ((energy * weighted).sum(-1) / den[..., 0] - center.square().sum(-1)).clamp_min(0)
        ds = torch.where(mask.sum(-1) == 1, torch.zeros_like(ds), ds)
        da = ((energy * cost).sum(-1) / cost.sum() - total.square().sum(-1)).clamp_min(0)
        std, mean = torch.std_mean(score, -1, unbiased=False, keepdim=True)
        std = std.clamp_min(1e-6)
        count = mask.sum(-1, keepdim=True)
        mz = (((score - mean) / std) * mask).sum(-1, keepdim=True) / count
        stats = torch.cat((total, center, ds[..., None], da[..., None], mean, std,
                           den, torch.ones_like(den) * cost.sum(),
                           (weighted * exact_d['origins']).sum(-1, keepdim=True), count, mz,
                           torch.zeros((*score.shape[:2], 7), device=q.device)), -1)
        indices = torch.cat((exact_d['rem'], exact_d['add']), -1).long()
        cv = content.gather(2, indices.clamp_min(0)[..., None].expand(-1, -1, -1, 32))
        cv = torch.where(indices[..., None] >= 0, cv, 0.)
        return dict(exact_d, score=score, stats=align(stats, exact_d['stats']), cv=align(cv, exact_d['cv']))

    @torch.no_grad()
    def deployed(self, q, raw, step):
        d = self.deployed_inputs(q, raw, step)
        mask = apply(d, exact(self.head, d).argmax(-1))
        return attend(self.variant, q, raw, mask).float(), mask, d

    def forward(self, q, raw, step, gate_temperature=.25, action_temperature=.25):
        with torch.no_grad():
            y, mask, exact_d = self.deployed(q, raw, step)
            hard_logits = exact(self.head, exact_d)
            logz, value = page_statistics(q, raw)
        d = self.differentiable_inputs(q, raw, exact_d)
        a = actions(d)
        approximate = surrogate(self.head, d)
        valid = a['valid']
        logits = align(torch.where(valid, approximate, 0.), torch.where(valid, hard_logits, 0.))
        probs = logits.masked_fill(~valid, -torch.inf).div(action_temperature).softmax(-1)
        std, mean = torch.std_mean(d['score'], -1, unbiased=False, keepdim=True)
        soft_base = (((d['score'] - mean) / std.clamp_min(1e-6) - self.beta) / gate_temperature).sigmoid()
        base = align(soft_base, d['base_mask'].float())
        removals = torch.zeros_like(base).scatter_add(-1, a['remove'].long().clamp_min(0), probs * (a['remove'] >= 0))
        additions = torch.zeros_like(base).scatter_add(-1, a['add'].long().clamp_min(0), probs * (a['add'] >= 0))
        soft_gate = base * (1. - removals) + additions
        gate = align(soft_gate, mask.float())
        actual = ActualAttention.apply(gate, y, logz, value)
        keep = (gate * d['costs']).sum(-1).mean() / (d['history_tokens'] + 4680)
        entropy = -(probs * probs.clamp_min(1e-20).log()).sum(-1).mean()
        return actual, keep, entropy
