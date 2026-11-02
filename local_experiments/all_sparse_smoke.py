"""Audit all sparse parameter families under a real frozen-backbone loss."""
import argparse,json,torch
from pathlib import Path
from .broad_common import Engine,noise
from .rollout_suite import generate_chunk
from .all_sparse_runtime import AllSparseRuntime
from .high_sparse_grad_smoke import timed
from .learned_cache.io import atomic
ROOT=Path('artifacts/sparse_expansion_20261031')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--chunk',type=int,default=2);args=ap.parse_args()
    assert json.loads((ROOT/'all_sparse_backward_checks.json').read_text())['passed']
    torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False
    rt=AllSparseRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt');emb=torch.load('artifacts/broad_coverage_20261030/embeddings/city_walk.pt',weights_only=True).cuda()
    try:
        for c in range(args.chunk):generate_chunk(engine,emb,0,c)
        x=noise(0,args.chunk,0);hard,_=timed(lambda:engine.forward(x,emb,args.chunk,0));n=list(rt.n);commits=list(rt.commits)
        rt.sparse=False;teacher,_=timed(lambda:engine.forward(x,emb,args.chunk,0));rt.sparse=True;rt.training_event=True
        torch.cuda.reset_peak_memory_stats()
        with torch.enable_grad():
            pred,fs=timed(lambda:Engine.forward.__wrapped__(engine,x,emb,args.chunk,0))
            fidelity=float((pred.detach().float()-hard.float()).square().sum()/hard.float().square().sum().clamp_min(1e-8));assert fidelity<1e-7,fidelity
            loss=(pred.float()-teacher.float()).square().mean()/teacher.float().square().mean().clamp_min(1e-8)
            _,bs=timed(loss.backward)
        rows=[];before={name:p.detach().clone() for name,p in rt.model.named_parameters()}
        for name,p in rt.model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
            rows.append(dict(name=name,gradient_norm=float(p.grad.norm())))
        opt=torch.optim.Adam(rt.model.parameters(),lr=1e-4);opt.step()
        assert all(not torch.equal(p,before[name]) for name,p in rt.model.named_parameters())
        assert all(p.grad is None and not p.requires_grad for p in engine.wrapper.parameters())
        assert rt.n==n and rt.commits==commits
        result=dict(passed=True,parameters=rows,loss=float(loss.detach()),hard_forward_nmse=fidelity,forward_seconds=fs,backward_seconds=bs,peak_allocated_bytes=torch.cuda.max_memory_allocated(),runtime=rt.finish())
        atomic(ROOT/f'all_sparse_smoke_chunk{args.chunk}.json',result);print(json.dumps(result),flush=True)
    finally:rt.training_event=False;rt.close()
if __name__=='__main__':main()
