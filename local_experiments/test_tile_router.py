"""Real replay smoke test for anchored initialization and tile objectives."""
import torch
from .tile_common import ROOT,cases,Engine
from .tile_router import Events,make_model,objective,evaluate_event,METHODS
from .learned_cache.io import atomic
from .broad_common import ROOT as DATA

def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();events=Events([0,17]);events.training=True;rows=[]
    emb=torch.load(DATA/'embeddings'/f'{cases("val")[0]["name"]}.pt',weights_only=True).cuda()
    def callback(current,q,k,v):
        if current['layer'] not in events.layers:return
        e=events.event(current,q,k,v)
        assert e['q'].shape==(12,32,128)
        routes=e['route'].reshape(12,8,4,128)
        assert torch.equal(routes,routes[:,:,:1].expand_as(routes))
        for method in METHODS:
            model=make_model(current['layer'],method).cuda()
            if method.startswith('anchor'):
                proto=model(e['h'])
                for a,b in zip(proto,e['h']['mean']):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
                baseline=evaluate_event(None,e)['nmse'];initial=evaluate_event(model,e)['nmse']
                assert max(abs(a-b) for a,b in zip(baseline,initial))<1e-7
                # Changing one physical page cannot alter any other page prototype.
                h=dict(e['h']);h['k']=h['k'].clone();h['k'][:,64:128]+=1
                changed=model(h)
                for a,b in zip(proto,changed):
                    torch.testing.assert_close(a[:,2:],b[:,2:],rtol=0,atol=0)
                    torch.testing.assert_close(a[:,:1],b[:,:1],rtol=0,atol=0)
            with torch.enable_grad():
                loss=objective(model,e,method);loss.backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
                norm=sum(g.square().sum().item() for g in grads)**.5
                assert norm>0
            rows.append(dict(layer=current['layer'],method=method,loss=loss.item(),grad_norm=norm))
    for chunk in range(6):
        data=torch.load(DATA/'trajectories'/cases('val')[0]['id']/f'chunk{chunk:02d}.pt',weights_only=True)
        if chunk==5:
            engine.callback=callback;engine.forward(data['noisy'][0].cuda(),emb,chunk,0);engine.callback=None
        engine.forward(data['clean'].cuda(),emb,chunk,4)
    assert len(rows)==8
    atomic(ROOT/'router_checks.json',dict(passed=True,rows=rows))
    print('Real replay: anchor=mean, page locality, shared tile route, finite nonzero gradients passed',flush=True)
if __name__=='__main__':run()
