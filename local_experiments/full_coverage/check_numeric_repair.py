"""Validate the actual failure and a dominant-page proxy stress case."""
import copy,json
import numpy as np
import torch
from .common import *
from .train import Bundle,load
from .training_execution import step
from .stable_targets import true_gains,proxy_gains,pages
from ..e24_joint_representatives.core import summary,contributions,route,gains,objective,predicted

def run():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rec=torch.load(OUT/'execution_optimization/nonfinite/s17_failure_state.pt',map_location='cpu',weights_only=False)
    b=Bundle(17);b.load_state_dict(rec['model']);opt=torch.optim.Adam([p for p in b.parameters() if p.requires_grad],lr=1e-4);opt.load_state_dict(rec['optimizer'])
    d=load(OUT/rec['event']);m=b.layers[d['key']];s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale']);qs=torch.linspace(0,63,16,device='cuda').round().long()
    with torch.no_grad():
        mask,r,a=route(d['q'][qs],s,m,d['own']);old=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);fixed=true_gains(c,d,s,m,mask,r,a);bad=~torch.isfinite(old)
        assert int(bad.sum())==16 and torch.isfinite(fixed).all() and torch.equal(old[~bad],fixed[~bad])
        # Independent CPU positive-mixture check for each repaired action.
        exact,approx=pages(c,d,s,m);ee=exact.cpu();ap=approx.cpu();le=d['logZ'].cpu();lm=c['logzh'].cpu();dense=d['dense'].cpu();base=objective(c,d['N'],d['Z'],d['dense'],mask).cpu();norm=dense.square().sum()+1e-12;errors=[]
        for j in bad.nonzero().flatten().tolist():
            ms=mask.cpu().clone();ms[int(r[j])]=False;ms[int(a[j])]=True
            weights=torch.where(ms[None],le,lm).softmax(-1);out=(weights[...,None]*torch.where(ms[None,:,None],ee,ap)).sum(1)
            truth=base-(out-dense).square().sum()/norm;errors.append(float((truth-fixed[j].cpu()).abs()))
        assert max(errors)<1e-12
    rng=np.random.default_rng(17);rng.bit_generator.state=copy.deepcopy(rec['rng']);row,_,_=step(b,opt,d,qs,B=torch.load(E24/'fit.pt',weights_only=True)['B'].cuda(),fit=torch.load(E24/'fit.pt',weights_only=True),u=rec['update'],rng=rng,reuse_risk=True,stable=True)
    assert all(torch.isfinite(p).all() for p in b.parameters()) and 'numeric_repairs' in row
    # Force beta40 and a single dominant summary page: the old subtraction denominator vanishes.
    rr=copy.deepcopy(b.reader);rr.page[-1].weight.data[16::17].zero_();rr.page[-1].bias.data[16::17].zero_();rr.page[-1].bias.data[16]=40.
    fake=dict(logit=torch.full((16,410,4),-1000.,device='cuda',dtype=torch.float64));fake['logit'][:,0]=0
    fake['logzh']=fake['logit'].logsumexp(-1)+np.log(16);fake['A']=fake['logzh'].exp().sum(1);fake['mass']=(fake['logzh']-fake['A'].log()[:,None]).exp()
    mk=torch.arange(410,device='cuda')<41;rem=torch.arange(16,device='cuda').repeat_interleave(32);add=torch.arange(41,73,device='cuda').repeat(16)
    z,uv,beta=rr(d['q'][qs],s,fake)
    fixed_proxy,used=proxy_gains(rr,d['q'][qs],s,fake,mk,rem,add,z,uv,beta)
    assert used and torch.isfinite(fixed_proxy).all()
    report=dict(passed=True,event=rec['event'],update=rec['update'],repaired_actions=16,finite_actions_bitwise_unchanged=496,cpu_reference_max_abs=max(errors),gradient_step_finite=True,proxy_dominant_page_finite=True,step=row,no_clipping=True,no_skipped_events=True,source=digest(__file__))
    atomic(OUT/'execution_optimization/nonfinite/repair_validation.json',report);print(json.dumps(report))
if __name__=='__main__':run()
