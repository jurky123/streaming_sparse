import math
import torch
import torch.nn.functional as F
from ..router.clustering import soft_kmeans
from ..router.budget_selection import select_under_budget


def spatial_assignment(positions, target_size=64):
    side = max(1, math.isqrt(target_size))
    cells = positions.long().clone()
    cells[:, 1:] //= side
    _, labels = torch.unique(cells, dim=0, sorted=True, return_inverse=True)
    return F.one_hot(labels).float()


@torch.no_grad()
def baseline_assignment(name, k, v, positions, target_size=64, max_blocks=256):
    if name == 'B0':
        return spatial_assignment(positions, target_size)
    features = k.float() if name == 'B1' else torch.cat((k.float(), v.float()), -1)
    features = features / features.shape[-1] ** 0.5
    return soft_kmeans(features, target_size=target_size, max_blocks=max_blocks)[1]


def mass_selection(stats, costs, budget, alpha=0.0):
    # Normalize each query before summing; per-query exponential scales differ.
    mass = (stats.z / (stats.z.sum(-1) + stats.z_current)[:, None].clamp_min(1e-30)).sum(0)
    ranks = mass / costs.clamp_min(1).pow(alpha)
    return select_under_budget(ranks, costs, budget)


@torch.no_grad()
def greedy_oracle(stats, costs, budget, target):
    from ..attention.reconstruction_loss import normalized_attention_mse
    mask = torch.zeros_like(costs)
    remaining = int(budget)
    while True:
        best, best_error = None, float('inf')
        for i in range(len(costs)):
            if mask[i] or not 0 < int(costs[i]) <= remaining:
                continue
            candidate = mask.clone()
            candidate[i] = 1
            error = float(normalized_attention_mse(stats.output(candidate), target))
            if error < best_error:
                best, best_error = i, error
        if best is None:
            break
        mask[best] = 1
        remaining -= int(costs[best])
    return mask
