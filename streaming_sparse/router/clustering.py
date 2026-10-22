import math
import torch
import torch.nn.functional as F


def squared_distance(x, centers):
    return (x.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None]
            - 2 * x @ centers.T).clamp_min(0)


def repair_labels(labels, distances, max_size):
    """Deterministic hard repair; the soft assignment supplies the STE derivative.

    Only empty/oversized clusters are repaired. No equal-size constraint is imposed.
    """
    labels = labels.clone()
    b = distances.shape[1]
    # Keep closest members within each oversized cluster. Repair all overflow in
    # one tensor operation rather than synchronizing CUDA once per moved token.
    overflow = []
    for block in range(b):
        indices = torch.where(labels == block)[0]
        if len(indices) > max_size:
            order = torch.argsort(distances[indices, block], stable=True)
            overflow.append(indices[order[max_size:]])
    if overflow:
        overflow = torch.cat(overflow)
        labels[overflow] = b
        counts = torch.bincount(labels, minlength=b+1)[:b]
        slots = torch.repeat_interleave(torch.arange(b,device=labels.device), max_size-counts)
        labels[overflow] = slots[:len(overflow)]
    counts = torch.bincount(labels, minlength=b)
    empty = torch.where(counts == 0)[0]
    if len(empty):
        # Leave one member in each donor cluster; donate distinct tokens to all
        # empty centers at once. This repair is exceptional and deterministic.
        donor_order = torch.argsort(labels, stable=True)
        sorted_labels = labels[donor_order]
        redundant = torch.cat((torch.zeros(1,dtype=torch.bool,device=labels.device),
                               sorted_labels[1:] == sorted_labels[:-1]))
        candidates = donor_order[redundant]
        own_distance = distances[candidates, labels[candidates]]
        tokens = candidates[torch.argsort(own_distance,descending=True,stable=True)[:len(empty)]]
        labels[tokens] = empty
    return labels


def soft_kmeans(features, blocks=None, temperature=0.1, iterations=3,
                target_size=64, max_blocks=256, max_size_factor=4):
    n = len(features)
    if n == 0:
        raise ValueError('History must contain at least one token')
    b = min(n, blocks or min(max_blocks, math.ceil(n / target_size)))
    # Fixed, evenly spaced token initialization; no random cluster IDs or ID embeddings.
    centers = features[torch.linspace(0, n - 1, b, device=features.device).long()]
    for _ in range(iterations):
        p = (-squared_distance(features, centers) / temperature).softmax(-1)
        centers = (p.T @ features) / p.sum(0)[:, None].clamp_min(1e-8)
    distances = squared_distance(features, centers)
    p = (-distances / temperature).softmax(-1)
    with torch.no_grad():
        labels = repair_labels(p.argmax(-1), distances.detach(),
                               max(1, math.ceil(max_size_factor * n / b)))
    hard = F.one_hot(labels, b).to(p.dtype)
    assignment = hard + (p - p.detach())
    # Mild collapse protection; uniform usage is NOT the training objective.
    usage = p.mean(0)
    regularizer = (usage * (usage.clamp_min(1e-8) * b).log()).sum()
    return assignment, hard, p, regularizer
