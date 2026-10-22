"""Minimal interventions over the frozen v0.3 router."""
import torch
from torch import nn
from ..model import SparseRouter


class IndependentScore(nn.Module):
    """Effective full pair weights, initialized exactly from the original score.

    For a 128->1 linear, rank-one residuals already span every weight vector.
    This intervention changes sharing/optimization, not linear expressivity.
    """
    def __init__(self, original, pairs):
        super().__init__()
        weights = original.base.weight.detach().expand(pairs, -1, -1).clone()
        if original.delta_up is not None:
            weights += original.delta_up.detach() @ original.delta_down.detach()
        self.weight = nn.Parameter(weights)
        self.bias = nn.Parameter(original.base.bias.detach().expand(pairs, -1).clone())

    def forward(self, x, pair):
        return torch.nn.functional.linear(x, self.weight[pair], self.bias[pair])


class RepairRouter(SparseRouter):
    def configure(self, arm):
        if arm not in ('A', 'B', 'C', 'D'):
            raise ValueError(arm)
        self.arm = arm
        if arm == 'A':
            self.features.requires_grad_(False)
        if arm == 'D':
            self.decoder.score = IndependentScore(self.decoder.score,
                                                 len(self.config.layers)*len(self.config.heads))
        return self

    def fixed_precompute(self, event, frozen):
        if self.arm not in ('A', 'B'):
            raise ValueError('Only A/B use the fixed partition')
        pair = self.pair_index(event['head_id'], event['layer_id'])
        features = (frozen['features'] if self.arm == 'A' else
                    self.features(torch.cat((event['k_hist'].float(), event['v_hist'].float()), -1), pair))
        hard = frozen['hard']
        z, empty = self.aggregator(features, hard, hard, pair)
        h = self.encoder(z, empty, pair)
        return dict(features=features, assignment=hard, hard=hard, probabilities=frozen['probabilities'],
                    raw_labels=frozen['raw_labels'], h=h, empty=empty, pair=pair,
                    layer_id=int(event['layer_id']), head_id=int(event['head_id']),
                    usage_diagnostic=frozen['usage_diagnostic'])
