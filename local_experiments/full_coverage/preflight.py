"""Disposable exact C update at covered and previously uncovered positions."""
import json
import numpy as np
import torch
from torch import nn
from .common import *
from .train import Bundle,load
from ..e24_joint_representatives.core import summary,contributions,labels,route,gains,objective,predicted
from ..learned_cache.io import atomic

def run():
    torch.set_num_threads(2);torch.manual_seed(17);bundle=Bundle(17)
    base=torch.load(PRE,map_location='cpu',weights_only=False)['model'];checks=[]
    for l in LAYERS:
        for h in HEADS:
            m=bundle.layers[f'l{l}h{h}']
            for n in ('bank','group_v','group_gate','pool_k','pool_v','pool_feature','basis'):
                assert torch.equal(getattr(m,n).cpu(),base[f'layers.{l}.{n}'][h:h+1]),(l,h,n)
            assert {n for n,p in m.named_parameters() if p.requires_grad}=={'pool_k','pool_v','pool_feature'}
    fit=torch.load(E24/'fit.pt',weights_only=True);B=fit['B'].cuda();ix=torch.linspace(0,63,16,device='cuda').round().long()
    optimizer=torch.optim.Adam([p for p in bundle.parameters() if p.requires_grad],lr=1e-4)
    for l,h in [(1,0),(2,1),(23,11)]:
        path=next((OUT/'data/city_walk_s701').glob(f'c2_l{l}_s0_h{h}_t*.pt'))
        d=load(path);m=bundle.layers[d['key']];s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale'])
        mask,r,a=route(d['q'][ix],s,m,d['own']);cp={n:v[ix] for n,v in c.items()};z,uv,beta=bundle.reader(d['q'][ix],s,cp)
        zt,ut,bt,_=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ']);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a).detach()
        approx=objective(c,d['N'],d['Z'],d['dense'],mask)
        res=nn.functional.smooth_l1_loss(z/fit['zscale'],zt[ix].detach()/fit['zscale'])+nn.functional.smooth_l1_loss(beta/fit['bscale'],bt[ix].detach()/fit['bscale'])
        pg=predicted(z,uv,cp['mass'],mask,r,a,beta)/(d['dense'].square().sum()/64);gs=fit['gscale']
        target=torch.softmax(torch.cat((g,g.new_zeros(1)))/(.3*gs),0)
        sl=nn.functional.smooth_l1_loss(pg/gs,g/gs)-.1*(target*torch.log_softmax(torch.cat((pg,pg.new_zeros(1)))/gs,0)).sum()
        loss=approx+.1*res+.1*sl;assert torch.isfinite(loss)
        before={n:getattr(m,n).detach().clone() for n in ('pool_k','pool_v','pool_feature')}
        optimizer.zero_grad();loss.backward();grad={n:float(getattr(m,n).grad.norm()) for n in before}
        assert all(np.isfinite(v) and v>0 for v in grad.values())
        torch.nn.utils.clip_grad_norm_([p for p in bundle.parameters() if p.requires_grad],1,error_if_nonfinite=True);optimizer.step()
        assert all(not torch.equal(getattr(m,n),v) for n,v in before.items())
        checks.append(dict(position=d['key'],event=str(path.relative_to(OUT)),loss=float(loss),pool_gradient_norms=grad,actions=len(r),own_protected=bool(mask[d['own']])))
    bundle.check_frozen()
    atomic(OUT/'joint_preflight.json',dict(passed=True,positions=348,initialization_exact=True,checks=checks,frozen_exact=True,updates_discarded=True,test_accessed=False))
    print(json.dumps(checks))
if __name__=='__main__':run()
