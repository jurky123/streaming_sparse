"""Check the actual finite-forward/overflowing-backward failure, without clipping targets."""
import argparse,copy,json
import numpy as np
import torch
from .common import *
from .train import Bundle,load
from .training_execution import step
from .stable_targets import proxy_gains
from ..e24_joint_representatives.core import summary,contributions,route,predicted

def run(seed=17):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    dest=OUT/'execution_optimization/gradient_failure';rec=torch.load(dest/f's{seed}_failure_state.pt',map_location='cpu',weights_only=False)
    b=Bundle(seed);b.load_state_dict(rec['model']);opt=torch.optim.Adam([p for p in b.parameters() if p.requires_grad],lr=1e-4);opt.load_state_dict(rec['optimizer'])
    d=load(OUT/rec['event']);m=b.layers[d['key']];s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale']);qs=torch.linspace(0,63,16,device='cuda').round().long()
    with torch.no_grad():
        mask,r,a=route(d['q'][qs],s,m,d['own']);cp={n:v[qs] for n,v in c.items()};z,uv,beta=b.reader(d['q'][qs],s,cp);old=predicted(z,uv,cp['mass'],mask,r,a,beta)
        fixed,used=proxy_gains(b.reader,d['q'][qs],s,cp,mask,r,a,z,uv,beta);assert used and torch.isfinite(fixed).all()
        # Independent CPU positive-denominator mixture for all 512 actions.
        mass=cp['mass'].cpu();exact=mass*beta.cpu().double().exp();zz=z.cpu();mk=mask.cpu();rr=r.cpu();aa=a.cpu();current=torch.where(mk[None],exact,mass).sum(1)
        H=zz[:,mk].sum(1)/current[:,None];target=zz.sum(1)/exact.sum(1)[:,None];E=target-H;reference=[]
        for j in range(len(r)):
            ms=mk.clone();ms[rr[j]]=False;ms[aa[j]]=True;den=torch.where(ms[None],exact,mass).sum(1);out=zz[:,ms].sum(1)/den[:,None];delta=out-H
            reference.append((2*(E*delta).sum(-1)-delta.square().sum(-1)).mean())
        reference=torch.stack(reference);error=float((fixed.cpu()-reference).abs().max())
        assert torch.allclose(fixed.cpu(),reference,atol=1e-22,rtol=1e-6),(error,float(reference.abs().max()))
    rng=np.random.default_rng(seed);rng.bit_generator.state=copy.deepcopy(rec['rng']);fit=torch.load(E24/'fit.pt',weights_only=True)
    row,_,_=step(b,opt,d,qs,fit['B'].cuda(),fit,rec['update'],rng,reuse_risk=True,stable=True)
    assert all(torch.isfinite(p).all() for p in b.parameters() if p.requires_grad)
    report=dict(passed=True,event=rec['event'],update=rec['update'],old_proxy_max_abs=float(old.abs().max()),new_proxy_max_abs=float(fixed.abs().max()),cpu_reference_max_abs=error,gradient_update_finite=True,gradient_clipping_unchanged=1.0,no_skipped_event=True,no_target_clipping=True,row=row,numerics_code=digest(ROOT/'local_experiments/full_coverage/stable_targets.py'))
    atomic(dest/('repair_validation.json' if seed==17 else f's{seed}_repair_validation.json'),report);print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=17);a=p.parse_args();run(a.seed)
