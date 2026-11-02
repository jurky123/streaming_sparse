import torch,json
from .high_sparse_gain import targets,loss
from .high_sparse_learn import mixed,correct
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.manual_seed(91);h,t,q,p,d=2,3,4,96,16
    truth=torch.randn(h,t,q,p).softmax(-1);values=torch.randn(h,t,q,p,d);av=torch.randn(h,p,d)
    e=dict(truth=truth,values=values,approx=truth*torch.randn_like(truth).mul(.2).exp(),target=(truth[...,None]*values).sum(-2),mass=truth.mean(2),own=torch.zeros(h,t,dtype=torch.long))
    mask=torch.zeros(h,t,p,dtype=torch.bool);mask[:,:,:10]=True
    add,drop,gain,before=targets(e,av,mask)
    errors=[]
    for i,j in [(0,0),(3,4),(7,7)]:
        new=mask.clone();new.scatter_(-1,add[...,i,None],True);new.scatter_(-1,drop[...,j,None],False)
        y=mixed(e,av,new)[0];norm=e['target'].square().sum(-1).clamp_min(1e-8)
        exact=before-((y-e['target']).square().sum(-1)/norm).mean(2)
        torch.testing.assert_close(gain[...,i,j],exact,atol=2e-6,rtol=2e-5);errors.append(float((gain[...,i,j]-exact).abs().max()))
    assert not (drop==0).any() and (add>=10).all()
    feat=torch.randn(h,t,p,14)
    with torch.enable_grad():
        w=torch.zeros(h,14,requires_grad=True);v=loss(e,feat,w,add,drop,gain,before);v.backward();assert torch.isfinite(w.grad).all() and w.grad.abs().sum()>0
    new,_=correct(mask,torch.randn(h,t,p),e['own']);assert (new.sum(-1)==10).all() and new[...,0].all() and ((mask^new).sum(-1)<=2).all()
    result=dict(passed=True,pair_delta_max_abs_errors=errors,own_protected=True,fixed_budget=True,gradient_finite=True)
    atomic('artifacts/high_sparse_gain_20261031/checks.json',result);print(json.dumps(result))
if __name__=='__main__':main()
