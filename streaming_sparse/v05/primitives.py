"""Per-query statistics and the original budget surrogate, batched over queries."""
import torch
from ..attention.block_statistics import differentiable_block_stats


class QueryStatistics:
    def __init__(self, legacy):
        self.z, self.u = legacy.z, legacy.u
        self.z_current, self.u_current = legacy.z_current, legacy.u_current

    def output(self, mask):
        if mask.ndim == 1:
            mask = mask.expand(len(self.z), -1)
        if mask.shape != self.z.shape:
            raise ValueError('Mask must be [N_Q,B] or [B]')
        m = mask.float()
        denominator = self.z_current + (self.z*m).sum(-1)
        numerator = self.u_current + (self.u*m[..., None]).sum(1)
        return numerator / denominator[:, None].clamp_min(1e-30)

    def dense(self):
        return self.output(torch.ones_like(self.z))

    def to(self, device):
        from types import SimpleNamespace
        return QueryStatistics(SimpleNamespace(**{k: getattr(self,k).to(device) for k in
                               ('z','u','z_current','u_current')}))


def statistics(q, event, assignment):
    return QueryStatistics(differentiable_block_stats(q,event['k_hist'],event['v_hist'],assignment,
                                                    event['k_current'],event['v_current']))


@torch.no_grad()
def select(scores, costs, budget, block_ids=None):
    if scores.ndim != 2 or costs.ndim != 1 or scores.shape[1] != len(costs):
        raise ValueError('Expected scores[Q,B], costs[B]')
    if block_ids is None:
        block_ids = torch.arange(len(costs),device=scores.device)
    # Original block ID is the deterministic tie-breaker, invariant to sequence permutation.
    canonical = torch.argsort(block_ids, stable=True)
    order = canonical[torch.argsort(scores[:,canonical],descending=True,stable=True)]
    order = order.cpu().tolist()
    integer_costs = costs.detach().long().cpu().tolist()
    chosen_q, chosen_b = [], []
    for q, indices in enumerate(order):
        remaining = int(budget)
        for b in indices:
            cost = integer_costs[b]
            if 0 < cost <= remaining:
                chosen_q.append(q); chosen_b.append(b); remaining -= cost
    result = torch.zeros_like(scores)
    if chosen_q:
        result[torch.tensor(chosen_q,device=scores.device),torch.tensor(chosen_b,device=scores.device)] = 1
    return result


class BudgetGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, costs, budget, temperature):
        active = costs.detach() > 0
        interior = 0 < budget < float(costs.detach().sum())
        if not interior:
            p = active.to(scores.dtype).expand_as(scores).clone() if budget > 0 else torch.zeros_like(scores)
        else:
            lo = scores.masked_fill(~active,torch.inf).amin(-1,keepdim=True)-40*temperature
            hi = scores.masked_fill(~active,-torch.inf).amax(-1,keepdim=True)+40*temperature
            for _ in range(64):
                mid = (lo+hi)/2
                mass = (costs*torch.sigmoid((scores-mid)/temperature)*active).sum(-1,keepdim=True)
                above = mass > budget
                lo,hi = torch.where(above,mid,lo),torch.where(above,hi,mid)
            p = torch.sigmoid((scores-(lo+hi)/2)/temperature)*active
        ctx.save_for_backward(p,costs)
        ctx.temperature,ctx.interior = temperature,interior
        return p

    @staticmethod
    def backward(ctx, grad):
        p,costs = ctx.saved_tensors
        if not ctx.interior:
            return torch.zeros_like(p),torch.zeros_like(costs),None,None
        w = p*(1-p)/ctx.temperature
        ratio = (grad*w).sum(-1,keepdim=True)/(costs*w).sum(-1,keepdim=True).clamp_min(1e-12)
        return w*(grad-costs*ratio),(-p*ratio).sum(0),None,None


def budget_mask(scores, assignment, hard, budget, temperature=.2, block_ids=None):
    hard_mask = select(scores,hard.sum(0),budget,block_ids)
    soft = BudgetGate.apply(scores,assignment.sum(0),int(budget),float(temperature))
    return hard_mask+(soft-soft.detach()),hard_mask,soft


def mass_mask(stat, costs, budget, block_ids=None):
    # Per-query mass ranking. No Mass target participates in training.
    mass = stat.z/(stat.z.sum(-1)+stat.z_current)[:,None].clamp_min(1e-30)
    return select(mass,costs,budget,block_ids)
