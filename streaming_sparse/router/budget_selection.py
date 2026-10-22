import torch
import torch.nn.functional as F


def priority(scores, costs, alpha=0.0):
    return scores if alpha == 0 else F.softplus(scores) / costs.clamp_min(1).pow(alpha)


@torch.no_grad()
def select_under_budget(scores, costs, budget):
    """Stable ranking with skip-on-overflow; integer token budget is never exceeded."""
    mask = torch.zeros_like(scores)
    remaining = int(budget)
    order = torch.argsort(scores, descending=True, stable=True).cpu().tolist()
    integer_costs = costs.detach().to(device='cpu', dtype=torch.long).tolist()
    chosen = []
    for index in order:
        cost = integer_costs[index]
        if 0 < cost <= remaining:
            chosen.append(index)
            remaining -= cost
    if chosen:
        mask[torch.tensor(chosen,device=scores.device)] = 1
    return mask


class _BudgetGate(torch.autograd.Function):
    """Implicit derivative of sum(cost * sigmoid((score-t)/tau)) = budget.

    A plain bisection made of torch.where has an incorrect threshold derivative.
    Solve numerically in forward, then differentiate the budget constraint exactly.
    """
    @staticmethod
    def forward(ctx, scores, costs, budget, temperature):
        active = costs.detach() > 0
        if budget <= 0:
            p = torch.zeros_like(scores)
            interior = False
        elif budget >= float(costs.detach().sum()):
            p = active.to(scores.dtype)
            interior = False
        else:
            lo = scores[active].min() - 40 * temperature
            hi = scores[active].max() + 40 * temperature
            for _ in range(64):
                mid = (lo + hi) / 2
                mass = (costs * torch.sigmoid((scores - mid) / temperature) * active).sum()
                above = mass > budget
                lo = torch.where(above, mid, lo)
                hi = torch.where(above, hi, mid)
            p = torch.sigmoid((scores - (lo + hi) / 2) / temperature) * active
            interior = True
        ctx.save_for_backward(p, costs)
        ctx.temperature, ctx.interior = temperature, interior
        return p

    @staticmethod
    def backward(ctx, grad):
        p, costs = ctx.saved_tensors
        if not ctx.interior:
            return torch.zeros_like(p), torch.zeros_like(costs), None, None
        w = p * (1 - p) / ctx.temperature
        ratio = (grad * w).sum() / (costs * w).sum().clamp_min(1e-12)
        return w * (grad - costs * ratio), -p * ratio, None, None


def soft_budget_relaxation(scores, costs, budget, temperature=0.2):
    return _BudgetGate.apply(scores, costs, int(budget), float(temperature))


def budget_mask(scores, assignment, hard, budget, alpha=0.0, temperature=0.2):
    costs = assignment.sum(0)
    ranks = priority(scores, costs, alpha)
    hard_mask = select_under_budget(ranks, hard.sum(0), budget)
    soft_mask = soft_budget_relaxation(ranks, costs, budget, temperature)
    return hard_mask + (soft_mask - soft_mask.detach()), hard_mask, soft_mask
