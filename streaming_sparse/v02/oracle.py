"""Offline exact-statistics greedy references. Never used as training labels."""
import torch


def batch_error(stats,masks,target):
    den=stats.z_current[None]+masks@stats.z.T
    num=stats.u_current[None]+torch.einsum('mb,qbd->mqd',masks,stats.u)
    output=num/den[:,:,None].clamp_min(1e-30)
    return (output-target.float()[None]).square().sum((1,2))/(target.float().square().sum()+1e-8)


@torch.no_grad()
def forward_greedy(stats,costs,budget,target):
    mask=torch.zeros_like(costs)
    remaining=int(budget)
    while True:
        feasible=torch.where((mask==0)&(costs>0)&(costs<=remaining))[0]
        if not len(feasible):
            break
        candidates=mask[None].expand(len(feasible),-1).clone()
        candidates[torch.arange(len(feasible),device=mask.device),feasible]=1
        errors=batch_error(stats,candidates,target)
        index=int(errors.argmin())
        mask=candidates[index]
        remaining-=int(costs[feasible[index]])
    return mask


@torch.no_grad()
def refine_mass(stats,costs,budget,target,mass,max_rounds=4,candidate_batch=64):
    """Best improving single add/remove/swap under budget, warm-started at Mass.

    This strengthens the heuristic but still does not prove an optimal bound.
    Variable block sizes can leave different budget utilization; record all costs.
    """
    mask=mass.clone()
    current=float(batch_error(stats,mask[None],target)[0])
    for _ in range(max_rounds):
        selected=torch.where(mask>0)[0].cpu().tolist()
        omitted=torch.where(mask==0)[0].cpu().tolist()
        integer_costs=costs.long().cpu().tolist()
        used=int((mask*costs).sum())
        edits=[(i,None) for i in selected]
        edits.extend((None,j) for j in omitted if used+integer_costs[j]<=budget)
        edits.extend((i,j) for i in selected for j in omitted if used-integer_costs[i]+integer_costs[j]<=budget)
        best_mask,best=mask,current
        for begin in range(0,len(edits),candidate_batch):
            batch=edits[begin:begin+candidate_batch]
            candidates=mask[None].expand(len(batch),-1).clone()
            remove=[(row,i) for row,(i,j) in enumerate(batch) if i is not None]
            add=[(row,j) for row,(i,j) in enumerate(batch) if j is not None]
            if remove:
                rows,cols=zip(*remove)
                candidates[list(rows),list(cols)]=0
            if add:
                rows,cols=zip(*add)
                candidates[list(rows),list(cols)]=1
            errors=batch_error(stats,candidates,target)
            index=int(errors.argmin())
            value=float(errors[index])
            if value<best-1e-12:
                best_mask,best=candidates[index].clone(),value
        if best>=current-1e-12:
            break
        mask,current=best_mask,best
    return mask
