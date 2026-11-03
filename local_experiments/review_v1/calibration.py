"""Continuous robust residual fitting and cost-aware threshold calibration."""
import torch

def fit_theta(features,labels,initial=None,steps=200):
    """Convex nonnegative Huber fit, independently per head.

    Normalize columns AND targets. Solve four coefficients directly rather than
    moving softplus logits across orders of magnitude. All scaling is computed
    on training data only; return coefficients in the original physical units.
    """
    import numpy as np
    from scipy.optimize import lsq_linear, minimize
    assert features.shape[:-1]==labels.shape and features.shape[-1]==4
    assert torch.isfinite(features).all() and torch.isfinite(labels).all()
    assert (features>=0).all() and (labels>=0).all()
    h=features.shape[0];heads=[];coeff=[]
    init=features.new_tensor([1,1,1,.5]).expand(h,4) if initial is None else initial
    for head in range(h):
        x=features[head].detach().double().cpu().reshape(-1,4).numpy()
        y=labels[head].detach().double().cpu().flatten().numpy()
        xs=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-12)
        ys=max(float(np.mean(y)),1e-12);xn=x/xs;yn=y/ys
        def objective(w):
            residual=xn@w-yn;absr=np.abs(residual)
            value=np.mean(np.where(absr<=1,.5*residual**2,absr-.5))
            grad=xn.T@np.clip(residual,-1,1)/len(yn)
            return float(value),grad
        original=init[head].detach().double().cpu().numpy()*xs/ys
        seed=lsq_linear(xn,yn,bounds=(0,np.inf),tol=1e-10,max_iter=steps)
        fit=minimize(objective,seed.x,jac=True,bounds=[(0,None)]*4,
                     method='L-BFGS-B',options=dict(maxiter=steps,ftol=1e-12,gtol=1e-9,maxls=50))
        # A failed optimizer never silently replaces a better initializer.
        best=min((seed.x,fit.x),key=lambda w:objective(w)[0])
        theta=np.maximum(best*ys/xs,1e-30)
        deployed=theta*xs/ys;value,grad=objective(deployed)
        projected=np.where((deployed<=1e-8)&(grad>0),0,grad)
        heads.append(dict(initial_loss=objective(original)[0],final_loss=value,
                          projected_gradient_inf=float(np.max(np.abs(projected))),
                          optimizer_success=bool(fit.success),iterations=int(fit.nit),
                          feature_rms=xs.tolist(),target_scale=ys,
                          prediction_to_label=float(np.mean(x@theta)/ys)))
        coeff.append(theta)
    theta=torch.tensor(np.stack(coeff),device=features.device,dtype=features.dtype)
    return theta,dict(initial_loss=sum(r['initial_loss'] for r in heads)/h,
                      final_loss=sum(r['final_loss'] for r in heads)/h,
                      solver='column-normalized nonnegative Huber; NNLS then L-BFGS-B',
                      max_projected_gradient=max(r['projected_gradient_inf'] for r in heads),heads=heads)

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
