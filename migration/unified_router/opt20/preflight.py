import time
import torch
from .setup import ROOT
from .model import TileRouter
from ..common import config,atomic_json,load_raw,event_key,paired_seed
from ..opt11.data import dense
from ..opt12.precision import clip_grad_norm

def main():
    _,m=config(ROOT);paired_seed(20);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    row=next(r for r in m['rows'] if r['split']=='train' and r['history_tokens']==14040);layer=24
    q=torch.load(ROOT/'full_queries'/event_key(row)/f'layer{layer:02d}.pt',map_location='cuda',weights_only=True)['q']
    raw=load_raw(m['parent_experiment'],row,layer);targets=torch.tensor(row['query_indices'],device='cuda')
    truth=dense(q[:,targets],raw).float();results={}
    for variant in m['variants']:
        model=TileRouter(m,layer,variant).cuda()
        model.load_state_dict(torch.load(ROOT/f'initialization/layer{layer:02d}.pt',map_location='cuda',weights_only=True)['model'])
        with torch.no_grad():expected,d=model.deployed(q,raw,targets)
        opt=torch.optim.AdamW(model.routing.parameters(),lr=m['learning_rate_router'],weight_decay=0.)
        before={k:p.detach().clone() for k,p in model.routing.items()}
        t=time.perf_counter();actual,metrics=model(q,raw,targets)
        assert torch.equal(actual,expected),'Training forward differs from real kernel'
        loss=(actual-truth).square().sum()/truth.square().sum()
        loss.backward();grad={k:float(p.grad.double().norm()) if p.grad is not None else None for k,p in model.routing.items()}
        assert all(v is not None and v>0 for v in grad.values()),grad
        assert all(torch.isfinite(p.grad).all() for p in model.routing.values())
        norm=clip_grad_norm(model.routing.parameters(),1.);opt.step();torch.cuda.synchronize()
        assert all(not torch.equal(p,before[k]) for k,p in model.routing.items())
        with torch.no_grad():_,next_d=model.deployed(q,raw,targets)
        assert not torch.equal(d['kv'].state['key'],next_d['kv'].state['key']),'Stale parameter-derived cache'
        assert float(metrics['executed'])<=model.budget+.012
        results[variant]=dict(actual_forward_bit_identical=True,all9_parameter_groups_updated=True,gradient_norm=float(norm),gradients=grad,
            actual_nmse=float(loss),keep=float(metrics['keep']),executed=float(metrics['executed']),seconds=time.perf_counter()-t,
            peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        print(variant,results[variant],flush=True)
        del model,opt,actual,loss,d,next_d;torch.cuda.empty_cache()
    atomic_json(ROOT/'preflight.json',dict(passed=True,layer=layer,event=row['sample_id'],variants=results,full_tile32_backward=True,backbone_frozen=True))

if __name__=='__main__':main()
