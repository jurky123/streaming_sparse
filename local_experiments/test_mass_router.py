"""Real replay checks for teacher normalization, control parity and variance."""
import torch
from .mass_common import ROOT,Engine,cases
from .mass_router import Events,make_model,objective,evaluate_event,scores,METHODS
from .tile_router import evaluate_event as previous_eval,objective as previous_loss
from .learned_cache.io import atomic

def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();events=Events([0,17]);events.training=True;rows=[]
    def callback(current,q,k,v):
        if current['layer'] not in events.layers:return
        e=events.event(current,q,k,v)
        torch.testing.assert_close(e['shared_mass'].sum(-1),torch.ones(12,8,device='cuda'))
        full=e['z'].softmax(-1).reshape(12,8,4,361).mean(2)[...,:-1]
        torch.testing.assert_close(e['shared_mass'],full/full.sum(-1,keepdim=True),atol=0,rtol=0)
        mean_model=make_model(current['layer'],'mass_mean').cuda()
        base=scores(mean_model(e['h']),e['route'])
        for method in METHODS:
            model=make_model(current['layer'],method).cuda()
            now=evaluate_event(model,e);old=previous_eval(None,e)
            assert now['nmse']==old['nmse'],(current,method)
            if method=='mass_var':
                torch.testing.assert_close(scores(model(e['h']),e['route'],e['route_var'],model.alpha),base,atol=0,rtol=0)
                with torch.no_grad():model.alpha.fill_(1.)
                changed=scores(model(e['h']),e['route'],e['route_var'],model.alpha)
                assert torch.isfinite(changed).all() and (changed-base).abs().max()>1e-5
                with torch.no_grad():model.alpha.fill_(-1.)
                model.project();assert (model.alpha==0).all()
            with torch.enable_grad():
                loss=objective(model,e,method)
                if method=='control':torch.testing.assert_close(loss,previous_loss(model,e,'anchor_tile'),atol=0,rtol=0)
                loss.backward()
                assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
                norm=sum(p.grad.square().sum().item() for p in model.parameters())**.5
                assert norm>0
            rows.append(dict(layer=current['layer'],method=method,loss=loss.item(),grad_norm=norm,
                alpha_grad=None if method!='mass_var' else model.alpha.grad.cpu().tolist()))
    engine.replay(cases('val')[0],callback,stop_chunk=5)
    assert len(rows)==24
    atomic(ROOT/'router_checks.json',dict(passed=True,rows=rows))
    print('teacher normalization/control parity/zero-variance-coefficient parity/finite gradients passed',flush=True)
if __name__=='__main__':run()
