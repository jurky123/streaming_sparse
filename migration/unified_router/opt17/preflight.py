import json,time,torch
from .setup import ROOT
from ..common import config,load_raw,event_key,atomic_json,paired_seed
from ..opt11.data import dense
from ..opt12.precision import clip_grad_norm
from ..opt5.probe import capture,timing
from ..opt16.route import route as parent_route
from .route import route
from .model import JointGroupedRouter

def main():
    _,m=config(ROOT);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;paired_seed(17)
    torch._dynamo.config.cache_size_limit=128;torch._dynamo.config.accumulated_cache_size_limit=1024
    row=next(r for r in m['rows']if r['split']=='train'and r['history_tokens']==14040);layer=24
    q=torch.load(ROOT/'full_queries'/event_key(row)/f'layer{layer:02d}.pt',map_location='cuda',weights_only=True)['q']
    raw=load_raw(m['parent_experiment'],row,layer);targets=torch.tensor(row['query_indices'],device='cuda');truth=dense(q[:,targets],raw).float()
    rows={};budgets=[]
    for variant in m['variants']:
        model=JointGroupedRouter(m,layer,variant).cuda();cal=torch.load(ROOT/f'calibration/layer{layer:02d}.pt',map_location='cuda',weights_only=True)
        assert cal['split']=='train';model.load_state_dict(cal['trees'],strict=False)
        with torch.no_grad():
            expected,d=model.deployed(q,raw,targets)
            old=parent_route(d['packed_q'],d['kv'].state,.15,.35)
            new=route(d['packed_q'],d['kv'].state,.15,.35,representatives=4)
            assert torch.equal(old,new),'R4 regression vs Opt16'
        before={n:p.detach().clone()for n,p in model.routing.items()}
        opt=torch.optim.AdamW(model.routing.parameters(),lr=3e-5,weight_decay=0.)
        started=time.perf_counter();actual,metrics=model(q,raw,targets,m['gate_temperature'])
        assert torch.equal(actual,expected),'Forward must be the actual hard deployed kernel'
        loss=(actual-truth).square().sum()/truth.square().sum()
        reference=(expected-truth).square().sum()/truth.square().sum();assert torch.equal(loss,reference)
        loss.backward()
        grads={n:float(p.grad.double().norm())if p.grad is not None else None for n,p in model.routing.items()}
        assert all(v is not None and v>0 for v in grads.values()),grads
        assert all(torch.isfinite(p.grad).all()for p in model.routing.values())
        norm=clip_grad_norm(model.routing.parameters(),1.);opt.step()
        changed={n:not torch.equal(p,before[n])for n,p in model.routing.items()};assert all(changed.values()),changed
        with torch.no_grad():
            _,next_d=model.deployed(q,raw,targets)
            assert not torch.equal(next_d['kv'].state['key'],d['kv'].state['key']),'stale prototype cache'
        rows[variant]=dict(forward_bit_identical=True,loss_bit_identical=True,all_routing_groups_finite_nonzero_gradient=True,gradients=grads,all_routing_parameters_changed=changed,gradient_norm=float(norm),actual_nmse=float(loss),keep=float(metrics['keep']),executed=float(metrics['executed']),seconds=time.perf_counter()-started,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        print('gradient_check',variant,rows[variant],flush=True)
        if variant=='Fast35':
            # Independently vary representatives at a fixed candidate and keep budget.
            for reps in [4,8,16]:
                with torch.no_grad():
                    y,dd=model.deployed(q,raw,targets,fraction=.35,representatives=reps)
                    fn=lambda:route(dd['packed_q'],dd['kv'].state,.15,.35,representatives=reps)
                    graph,_=capture(fn);timing(graph,100)
                    budgets.append(dict(representatives=reps,candidate_fraction=.35,route_ms=timing(graph,100),actual_nmse=float((y-truth).square().sum()/truth.square().sum()),executed=float(dd['executed'])))
        del model,actual,expected,d,next_d,loss,opt;torch.cuda.empty_cache()
    atomic_json(ROOT/'preflight.json',dict(passed=True,layer=layer,event=row['sample_id'],split='train',variants=rows,representative_probe=budgets,full_q_context=4680,loss_queries=len(targets),backbone_frozen=True,trees_update_alternately=True))
    print('PREFLIGHT PASSED',budgets,flush=True)

if __name__=='__main__':main()
