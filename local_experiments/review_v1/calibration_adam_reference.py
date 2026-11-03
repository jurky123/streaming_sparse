"""Continuous robust residual fitting and cost-aware threshold calibration."""
import torch

def fit_theta(features,labels,initial=None,steps=200):
    """H,Q,P,4 features and H,Q,P labels; call on TRAIN observations only."""
    assert features.shape[:-1]==labels.shape and features.shape[-1]==4
    h=features.shape[0]
    init=features.new_tensor([1,1,1,.5]).expand(h,4) if initial is None else initial
    raw=torch.nn.Parameter(init.clamp_min(1e-6).expm1().log().clone())
    opt=torch.optim.Adam([raw],lr=.05)
    scale=labels.mean((1,2),keepdim=True).clamp_min(1e-6)
    history=[]
    with torch.enable_grad():
        for _ in range(steps):
            pred=(features.detach()*torch.nn.functional.softplus(raw)[:,None,None]).sum(-1)
            loss=torch.nn.functional.smooth_l1_loss(pred/scale,labels.detach()/scale)
            opt.zero_grad();loss.backward();opt.step();history.append(float(loss.detach()))
    return torch.nn.functional.softplus(raw).detach(),dict(initial_loss=history[0],final_loss=history[-1],steps=steps)

@torch.no_grad()
def calibrate_threshold(score_batches,own_batches,config,iterations=48):
    """One layer/head/step's replay batches, each Qtiles,P (varying P).
    Protected page and cap included. Global count ratio, not mean density.
    Returns theta score threshold = lambda * constant exact-page cost.
    Caller must record warmup/full/eviction coverage and summary identity.
    """
    assert len(score_batches)==len(own_batches)>0
    denominator=sum(s.numel() for s in score_batches)
    target=config.density*denominator
    def count(threshold):
        total=0
        for scores,own in zip(score_batches,own_batches):
            selected=(scores>threshold).scatter(-1,own[:,None],True)
            total+=int(selected.sum(-1).clamp(max=__import__('math').ceil(config.max_fraction*scores.shape[-1])).sum())
        return total
    low=0.;high=max(float(s.max()) for s in score_batches)+1
    for _ in range(iterations):
        mid=(low+high)/2
        if count(mid)>target:low=mid
        else:high=mid
    chosen=min((low,high),key=lambda x:abs(count(x)-target));actual=count(chosen)
    return dict(threshold=chosen,lambda_flop=chosen/2097152,selected=actual,candidates=denominator,actual_density=actual/denominator,target_density=config.density,discrete_target_error=actual-target)
