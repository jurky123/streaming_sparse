"""The frozen v0.3 architecture, isolated from the reproducible v0.2 tree."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..router.clustering import soft_kmeans
from ..router.budget_selection import budget_mask, select_under_budget


@dataclass
class RouterConfig:
    version: str = '0.3'
    fusion: str = 'concat'
    conditioned: bool = True
    residual_rank: int = 1
    layers: tuple = (4, 14, 24)
    heads: tuple = tuple(range(12))
    head_dim: int = 128
    hidden: int = 128
    query_tile: int = 16
    target_size: int = 64
    max_blocks: int = 256
    cluster_iterations: int = 3
    tau_cluster: float = .1
    tau_mask: float = .2
    max_size_factor: float = 4.

    def __post_init__(self):
        self.layers, self.heads = tuple(self.layers), tuple(self.heads)
        if self.version != '0.3' or self.fusion not in ('concat', 'film'):
            raise ValueError('Expected v0.3 with concat or film fusion')
        if (self.head_dim, self.hidden, self.query_tile) != (128, 128, 16):
            raise ValueError('Frozen architecture requires head_dim=hidden=128 and query_tile=16')
        if not self.layers or not self.heads or len(set(self.layers)) != len(self.layers) or len(set(self.heads)) != len(self.heads):
            raise ValueError('Layer/head IDs must be nonempty and unique')
        if min(self.target_size, self.max_blocks, self.cluster_iterations, self.residual_rank) < 1:
            raise ValueError('Sizes, iterations, and residual rank must be positive')
        if min(self.tau_cluster, self.tau_mask) <= 0 or self.max_size_factor < 1:
            raise ValueError('Invalid clustering/gate temperatures or repair bound')


class PairLinear(nn.Module):
    """W(l,h) = W0 + U(l,h) V(l,h), with one independent pair index.

    Bias is shared. Both small factors start nonzero so both receive gradients
    at the first update. No full per-pair router or additive layer/head factors.
    """
    def __init__(self, n_in, n_out, pairs, rank):
        super().__init__()
        self.base = nn.Linear(n_in, n_out)
        if pairs:
            self.delta_down = nn.Parameter(torch.randn(pairs, rank, n_in) * .01)
            self.delta_up = nn.Parameter(torch.randn(pairs, n_out, rank) * .01)
        else:
            self.register_parameter('delta_down', None)
            self.register_parameter('delta_up', None)

    def forward(self, x, pair):
        y = self.base(x)
        if self.delta_down is not None:
            y = y + F.linear(F.linear(x, self.delta_down[pair]), self.delta_up[pair])
        return y


class PairNorm(nn.Module):
    def __init__(self, width, pairs):
        super().__init__()
        self.base = nn.LayerNorm(width)
        if pairs:
            self.delta_scale = nn.Parameter(torch.zeros(pairs, width))
            self.delta_shift = nn.Parameter(torch.zeros(pairs, width))
        else:
            self.register_parameter('delta_scale', None)
            self.register_parameter('delta_shift', None)

    def forward(self, x, pair):
        y = self.base(x)
        if self.delta_scale is not None:
            # Residual in affine norm parameters, not a second normalization.
            normalized = F.layer_norm(x, self.base.normalized_shape, eps=self.base.eps)
            y = y + normalized * self.delta_scale[pair] + self.delta_shift[pair]
        return y


class PairMLP(nn.Module):
    def __init__(self, widths, pairs, rank):
        super().__init__()
        self.linears = nn.ModuleList(PairLinear(a, b, pairs, rank) for a, b in zip(widths, widths[1:]))

    def forward(self, x, pair):
        for i, layer in enumerate(self.linears):
            x = layer(x, pair)
            if i < len(self.linears) - 1:
                x = F.gelu(x)
        return x


class SetEncoderLayer(nn.Module):
    """Pre-norm TransformerEncoderLayer: width 128, 4 heads, FFN 512.

    Explicit projections expose the same pair weight residual in every learned
    attention/FFN module. No position encoding; dropout=0 preserves cache identity.
    """
    def __init__(self, pairs, rank):
        super().__init__()
        self.norm1, self.norm2 = PairNorm(128, pairs), PairNorm(128, pairs)
        self.qkv = PairLinear(128, 384, pairs, rank)
        self.out = PairLinear(128, 128, pairs, rank)
        self.ffn = PairMLP((128, 512, 128), pairs, rank)

    def forward(self, x, empty, pair):
        q, k, v = self.qkv(self.norm1(x, pair), pair).chunk(3, -1)
        q, k, v = [t.reshape(len(t), 4, 32).transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
        allowed = (~empty)[None, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.)
        y = y.squeeze(0).transpose(0, 1).reshape(len(x), 128)
        x = x + self.out(y, pair)
        x = x + self.ffn(self.norm2(x, pair), pair)
        return x.masked_fill(empty[:, None], 0)


class BlockAggregator(nn.Module):
    def __init__(self, pairs, rank):
        super().__init__()
        self.transform = PairLinear(128, 128, pairs, rank)
        self.projection = PairLinear(384, 128, pairs, rank)

    def forward(self, features, assignment, hard, pair):
        content = F.gelu(self.transform(features, pair))
        costs = assignment.sum(0)
        mean = assignment.T @ content / costs[:, None].clamp_min(1e-8)
        labels = hard.argmax(-1)[:, None].expand_as(content)
        shape = (hard.shape[1], 128)
        maximum = content.new_full(shape, -torch.inf).scatter_reduce(0, labels, content, reduce='amax', include_self=True)
        minimum = content.new_full(shape, torch.inf).scatter_reduce(0, labels, content, reduce='amin', include_self=True)
        empty = hard.sum(0) == 0
        pooled = torch.cat((mean, maximum.masked_fill(empty[:, None], 0), minimum.masked_fill(empty[:, None], 0)), -1)
        return self.projection(pooled, pair).masked_fill(empty[:, None], 0), empty


class QueryDecoder(nn.Module):
    def __init__(self, fusion, pairs, rank):
        super().__init__()
        self.kind = fusion
        # Retain mean pooling followed by a two-linear query MLP.
        self.query = PairMLP((128, 256, 128), pairs, rank)
        self.fusion = PairLinear(384 if fusion == 'concat' else 128,
                                 128 if fusion == 'concat' else 256, pairs, rank)
        if fusion == 'film':
            with torch.no_grad():
                self.fusion.base.bias[:128].fill_(1.)
                self.fusion.base.bias[128:].zero_()
        self.post = nn.ModuleList(SetEncoderLayer(pairs, rank) for _ in range(2))
        self.score = PairLinear(128, 1, pairs, rank)

    def forward(self, q, h, empty, pair):
        q = self.query(q.float().mean(0), pair)
        if self.kind == 'concat':
            expanded = q.expand_as(h)
            x = F.gelu(self.fusion(torch.cat((h, expanded, h * expanded), -1), pair))
        else:
            gamma, beta = self.fusion(q, pair).chunk(2, -1)
            x = gamma * h + beta
        for layer in self.post:
            x = layer(x, empty, pair)
        return self.score(x, pair).squeeze(-1).masked_fill(empty, 0)


class SparseRouter(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RouterConfig()
        c = self.config
        pairs = len(c.layers) * len(c.heads) if c.conditioned else 0
        self.features = PairMLP((256, 256, 128, 128), pairs, c.residual_rank)
        self.aggregator = BlockAggregator(pairs, c.residual_rank)
        self.encoder = SetEncoderLayer(pairs, c.residual_rank)
        self.decoder = QueryDecoder(c.fusion, pairs, c.residual_rank)
        capacity = self.parameter_counts()
        if capacity['pair_specific'] >= capacity['shared']:
            raise ValueError(f'Pair residual capacity must be smaller than shared base: {capacity}')

    def pair_index(self, head_id, layer_id):
        try:
            return self.config.layers.index(int(layer_id)) * len(self.config.heads) + self.config.heads.index(int(head_id))
        except ValueError as error:
            raise ValueError(f'Unsupported layer/head pair ({layer_id}, {head_id})') from error

    def precompute(self, k, v, positions=None, head_id=0, layer_id=4):
        # positions is accepted for the existing adapter interface, never used.
        if k.ndim != 2 or k.shape != v.shape or k.shape[-1] != 128 or len(k) == 0:
            raise ValueError('Expected nonempty N x 128 history K/V')
        pair = self.pair_index(head_id, layer_id)
        features = self.features(torch.cat((k.float(), v.float()), -1), pair)
        c = self.config
        assignment, hard, probabilities, usage_diagnostic = soft_kmeans(
            features, temperature=c.tau_cluster, iterations=c.cluster_iterations,
            target_size=c.target_size, max_blocks=c.max_blocks, max_size_factor=c.max_size_factor)
        z, empty = self.aggregator(features, assignment, hard, pair)
        h = self.encoder(z, empty, pair)
        return dict(features=features, assignment=assignment, hard=hard, probabilities=probabilities,
                    raw_labels=probabilities.argmax(-1), h=h, empty=empty, pair=pair,
                    layer_id=int(layer_id), head_id=int(head_id), usage_diagnostic=usage_diagnostic)

    def route(self, q, state, keep_ratio, relaxation=True):
        if not 0 <= keep_ratio <= 1:
            raise ValueError('keep_ratio must be in [0,1]')
        if q.ndim != 2 or not 0 < len(q) <= self.config.query_tile or q.shape[-1] != 128:
            raise ValueError('Expected one query tile with 1..16 tokens and width 128')
        scores = self.decoder(q, state['h'], state['empty'], state['pair'])
        budget = math.floor(keep_ratio * len(state['hard']))
        if relaxation:
            mask, hard, soft = budget_mask(scores, state['assignment'], state['hard'], budget,
                                            temperature=self.config.tau_mask)
        else:
            hard = select_under_budget(scores, state['hard'].sum(0), budget)
            mask, soft = hard, None
        return dict(mask=mask, hard_mask=hard, soft_mask=soft, scores=scores, budget=budget)

    def parameter_counts(self):
        residual = sum(p.numel() for n, p in self.named_parameters() if 'delta_' in n)
        total = sum(p.numel() for p in self.parameters())
        return dict(shared=total-residual, pair_specific=residual, total=total,
                    residual_over_base=residual/max(total-residual, 1))

    def checkpoint(self):
        return dict(config=asdict(self.config), router=self.state_dict(), parameter_counts=self.parameter_counts())


def load_router(path, device='cpu'):
    value = torch.load(path, map_location=device, weights_only=True)
    if value.get('config', {}).get('version') != '0.3':
        raise ValueError('This loader requires a v0.3 router checkpoint; v0.2 stays separate')
    model = SparseRouter(RouterConfig(**value['config'])).to(device)
    model.load_state_dict(value['router'], strict=True)
    model.run_identity = value.get('run_identity')
    return model
