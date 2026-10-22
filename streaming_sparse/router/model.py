from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from .joint_feature import JointFeature
from .clustering import soft_kmeans
from .aggregator import BlockAggregator
from .encoder import BlockEncoder
from .decoder import QueryDecoder
from .budget_selection import budget_mask, priority, select_under_budget


@dataclass
class RouterConfig:
    head_dim: int = 128
    heads: int = 12
    layers: int = 30
    target_size: int = 64
    max_blocks: int = 256
    cluster_iterations: int = 3
    tau_cluster: float = 0.1
    tau_mask: float = 0.2
    max_size_factor: float = 4.0
    alpha: float = 0.0
    query_tile: int = 16
    grouping: str = 'learned'


class SparseRouter(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RouterConfig()
        c = self.config
        self.features = JointFeature(c.head_dim, c.heads, c.layers)
        self.aggregator = BlockAggregator()
        self.encoder = BlockEncoder()
        self.decoder = QueryDecoder(c.head_dim)

    def precompute(self, k, v, positions, head_id=0, layer_id=0):
        c = self.config
        g, content = self.features(k, v, positions, head_id, layer_id)
        if c.grouping == 'spatial':
            from ..evaluation.baselines import spatial_assignment
            a = hard = spatial_assignment(positions, c.target_size).to(k.device)
            p, reg = hard, g.sum() * 0
        else:
            a, hard, p, reg = soft_kmeans(g, temperature=c.tau_cluster,
                iterations=c.cluster_iterations, target_size=c.target_size,
                max_blocks=c.max_blocks, max_size_factor=c.max_size_factor)
        z, empty = self.aggregator(content, a, hard, positions)
        h = self.encoder(z, empty)
        return dict(assignment=a, hard=hard, probabilities=p, h=h, regularizer=reg)

    def route(self, q, state, keep_ratio, relaxation=True):
        if not 0 <= keep_ratio <= 1:
            raise ValueError('keep_ratio must be in [0,1]')
        c = self.config
        scores = self.decoder(q, state['h'])
        budget = math.floor(keep_ratio * state['hard'].shape[0])
        if relaxation:
            mask, hard, soft = budget_mask(scores, state['assignment'], state['hard'], budget,
                                           c.alpha, c.tau_mask)
        else:
            costs = state['hard'].sum(0)
            hard = select_under_budget(priority(scores,costs,c.alpha),costs,budget)
            mask, soft = hard, None
        return dict(mask=mask, hard_mask=hard, soft_mask=soft, scores=scores, budget=budget)

    def checkpoint(self):
        return {'config': asdict(self.config), 'router': self.state_dict()}


def load_router(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    router = SparseRouter(RouterConfig(**checkpoint['config'])).to(device)
    router.load_state_dict(checkpoint['router'])
    return router
