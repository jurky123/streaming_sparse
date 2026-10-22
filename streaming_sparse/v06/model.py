"""Exactly shared C0 base; independent local/attention/signed correction modules."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..v05.model import (RouterConfig as PreviousConfig, HistoricalFrontend, QueryFusion,
                          InteractionBlock, PairLinear, PairMLP, PairNorm)
from ..v05.primitives import budget_mask
from .common import tensor_digest


@dataclass
class RouterConfig(PreviousConfig):
    version: str = '0.6'

    def __post_init__(self):
        if self.version != '0.6':
            raise ValueError('Expected v0.6 configuration')
        # Reuse every architecture restriction without changing frozen v0.5 files.
        self.version = '0.5'
        super().__post_init__()
        self.version = '0.6'


class Relevance(nn.Module):
    def __init__(self, c):
        super().__init__()
        pairs = len(c.layers) * len(c.heads)
        self.size = PairLinear(1, 128, pairs, c.residual_rank)
        self.mlp = PairMLP((128, 128, 1), pairs, c.residual_rank)

    def forward(self, x, state):
        with torch.autocast(device_type=x.device.type, enabled=False):
            size = self.size(state['hard'].sum(0).log1p()[:, None], state['pair'])
            return self.mlp(x.float() + size[None], state['pair']).squeeze(-1).masked_fill(state['empty'][None], 0)


class LocalBlock(InteractionBlock):
    """Pointwise gated MLP: same parameters as attention, no peer block reads."""
    def __init__(self, c):
        super().__init__(c, signed=False)

    def forward(self, x, empty, pair, mode='normal', trace=False):
        a, b, v = self.qkv(self.norm1(x, pair), pair).chunk(3, -1)
        message = torch.sigmoid(a * b / math.sqrt(32)) * v
        projected = self.out(message, pair) - self.out(torch.zeros_like(message), pair)
        update = self.scale * projected
        y = x + update
        y = (y + self.ffn(self.norm2(y, pair), pair)).masked_fill(empty[None, :, None], 0)
        return y, projected, dict(update=update, message=message, relation=None, interaction_vector=None)


class SignedVariant(InteractionBlock):
    def __init__(self, c, variant):
        super().__init__(c, signed=True)
        self.variant = variant

    def forward(self, x, empty, pair, mode='normal', trace=False):
        if self.variant == 'C3':
            return super().forward(x, empty, pair, mode, trace)
        if self.variant == 'C3_nocenter':
            return super().forward(x, empty, pair, 'no_center', trace)
        if self.variant == 'C3_positive':
            return super().forward(x, empty, pair, 'positive', trace)
        if self.variant != 'C3_amplitude':
            raise ValueError(self.variant)
        normalized = self.norm1(x, pair)
        with torch.autocast(device_type=x.device.type, enabled=False):
            u = self.u_norm(self.u(normalized.float(), pair), pair)
            valid = (~empty).float()[None, :, None]
            u = (u - (u * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)) * valid
            relation = u @ u.transpose(-1, -2) / math.sqrt(u.shape[-1])
            allowed = (~empty)[:, None] & (~empty)[None, :] & ~torch.eye(len(empty), device=x.device, dtype=torch.bool)
            relation = relation.masked_fill(~allowed[None], 0)
            message = (relation @ self.value(normalized.float(), pair)) / (relation.abs().sum(-1, keepdim=True) + 1e-6)
            projected = self.out(message, pair) - self.out(torch.zeros_like(message), pair)
        update = self.scale * projected
        y = x + update
        y = (y + self.ffn(self.norm2(y, pair), pair)).masked_fill(empty[None, :, None], 0)
        info = dict(update=update, message=message)
        if trace:
            info.update(relation=relation, interaction_vector=u)
        return y, projected, info


class Correction(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        pairs = len(c.layers) * len(c.heads)
        constructor = (lambda: LocalBlock(c)) if kind == 'C1' else (
            (lambda: InteractionBlock(c, False)) if kind == 'C2' else (lambda: SignedVariant(c, kind)))
        self.interaction = nn.ModuleList(constructor() for _ in range(2))
        self.score_norm, self.message_norm = PairNorm(128, pairs), PairNorm(128, pairs)
        self.head = PairMLP((256, 128, 1), pairs, c.residual_rank)
        self.alpha = nn.Parameter(torch.tensor(.1))

    def forward(self, x, state, trace=False):
        y = x; messages = []; infos = []
        for layer in self.interaction:
            y, message, info = layer(y, state['empty'], state['pair'], trace=trace)
            messages.append(message); infos.append(info)
        with torch.autocast(device_type=x.device.type, enabled=False):
            m = torch.stack(messages).mean(0).float(); zero = torch.zeros_like(m)
            mn = self.message_norm(m, state['pair']) - self.message_norm(zero, state['pair'])
            yn = self.score_norm(y.float(), state['pair'])
            delta = (self.head(torch.cat((yn, mn), -1), state['pair']) -
                     self.head(torch.cat((yn, torch.zeros_like(mn)), -1), state['pair'])).squeeze(-1)
        return self.alpha * delta, dict(correction=delta, scaled_correction=self.alpha * delta,
                                      hidden=y, layers=infos)


KINDS = ('C1', 'C2', 'C3', 'C3_nocenter', 'C3_amplitude', 'C3_positive')


class Router(nn.Module):
    def __init__(self, config=None, arms=(), correction_seed=100000):
        super().__init__()
        self.config = config or RouterConfig()
        self.frontend = HistoricalFrontend(self.config)
        self.query_fusion = QueryFusion(self.config)
        self.relevance = Relevance(self.config)
        self.arms = tuple(arms)
        if not set(self.arms) <= set(KINDS):
            raise ValueError(self.arms)
        # fork_rng prevents correction constructors from changing fresh-C0 initialization/RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(correction_seed)
            templates = {kind: Correction(self.config, kind) for kind in KINDS}
            reference = templates['C2'].state_dict()
            signed_reference = templates['C3'].state_dict()
            for kind, module in templates.items():
                own = module.state_dict()
                for source in (reference, signed_reference if kind.startswith('C3') else {}):
                    own.update({k: v for k, v in source.items() if k in own and own[k].shape == v.shape})
                module.load_state_dict(own)
            self.branches = nn.ModuleDict({kind: templates[kind] for kind in self.arms})
        self.correction_seed = correction_seed
        self.base_frozen = False

    def base_items(self):
        return [(name, value) for name, value in self.state_dict().items() if not name.startswith('branches.')]

    def base_digest(self):
        return tensor_digest(self.base_items())

    def freeze_base(self):
        for module in (self.frontend, self.query_fusion, self.relevance):
            module.requires_grad_(False)
            module.eval()
        self.base_frozen = True

    def train(self, mode=True):
        super().train(mode)
        if self.base_frozen:
            for module in (self.frontend, self.query_fusion, self.relevance):
                module.eval()
        return self

    def precompute(self, k, v, positions=None, head_id=0, layer_id=4):
        return self.frontend(k, v, layer_id, head_id)

    def predict(self, q, state, trace=False, only=None):
        if q.ndim != 2 or q.shape[1] != 128 or not 1 <= len(q) <= 16:
            raise ValueError('Expected 1..16 independent raw queries of width128')
        with torch.autocast(device_type=q.device.type, dtype=torch.bfloat16,
                            enabled=self.config.bf16 and q.device.type == 'cuda'):
            x, e = self.query_fusion(q, state['h'], state['empty'], state['pair'])
        r = self.relevance(x, state)
        result = {'C0': dict(scores=r, relevance=r, correction=torch.zeros_like(r),
                            scaled_correction=torch.zeros_like(r), layers=[])}
        for name, branch in self.branches.items():
            if only is not None and name not in only:
                continue
            with torch.autocast(device_type=q.device.type, dtype=torch.bfloat16,
                                enabled=self.config.bf16 and q.device.type == 'cuda'):
                d, info = branch(x, state, trace)
            result[name] = dict(scores=(r + d).masked_fill(state['empty'][None], 0), relevance=r, **info)
        if trace:
            for route in result.values():
                route.update(fusion=x, query_embedding=e)
        return result

    def route(self, q, state, keep=.2, trace=False, predictions=None, only=None):
        if not 0 <= keep <= 1:
            raise ValueError(keep)
        budget = math.floor(keep * len(state['hard']))
        predictions = self.predict(q, state, trace, only) if predictions is None else predictions
        result = {}
        for name, prediction in predictions.items():
            with torch.autocast(device_type=q.device.type, enabled=False):
                mask, hard, soft = budget_mask(prediction['scores'].float(), state['assignment'], state['hard'],
                                                budget, self.config.tau_mask, state['block_ids'])
            result[name] = dict(prediction, mask=mask, hard_mask=hard, soft_mask=soft, budget=budget)
        return result

    def capacity(self):
        base = sum(p.numel() for module in (self.frontend, self.query_fusion, self.relevance) for p in module.parameters())
        arms = {k: sum(p.numel() for p in v.parameters()) for k, v in self.branches.items()}
        return dict(base=base, correction=arms, total_per_model={k: base + v for k, v in arms.items()},
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def checkpoint(self):
        return dict(config=asdict(self.config), arms=self.arms, correction_seed=self.correction_seed,
                    router=self.state_dict(), capacity=self.capacity(), base_digest=self.base_digest())


def load_router(path, device='cpu'):
    value = torch.load(path, map_location=device, weights_only=True)
    model = Router(RouterConfig(**value['config']), value['arms'], value['correction_seed']).to(device)
    model.load_state_dict(value['router'], strict=True)
    if model.base_digest() != value['base_digest']:
        raise ValueError('Checkpoint base digest mismatch')
    return model, value


def from_base(path, arms, seed, device='cpu'):
    base, saved = load_router(path, device)
    if base.arms:
        raise ValueError('StageII must start from independent C0, not a joint-training checkpoint')
    model = Router(base.config, arms, 100000 + seed).to(device)
    own = model.state_dict(); own.update(dict(base.base_items())); model.load_state_dict(own)
    if model.base_digest() != base.base_digest():
        raise AssertionError('Different C0 bases')
    return model, saved
