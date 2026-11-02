"""Real two-denoise-step gradient audit; backbone frozen, no cross-chunk BPTT."""
import json,time,types,torch
from torch.utils.checkpoint import checkpoint
from .broad_common import Engine,noise
from .rollout_suite import generate_chunk
from .all_sparse_runtime import AllSparseRuntime
from .learned_cache.io import atomic
ROOT='artifacts/sparse_expansion_20261031/'
def main():
    torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False
    rt=AllSparseRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt')
    # Each checkpoint recomputes under its own step context, even after step 2 ran.
    for i,b in enumerate(engine.wrapper.model.blocks):
        original=rt.original_blocks[i]
        def wrapped(module,*args,index=i,original=original,**kwargs):
            if not(rt.training_event and index>0 and torch.is_grad_enabled()):return original(*args,**kwargs)
            context=dict(engine.current)
            def replay(*xs,**kw):
                previous=dict(engine.current);engine.current.update(context)
                try:return original(*xs,**kw)
                finally:engine.current.clear();engine.current.update(previous)
            return checkpoint(replay,*args,**kwargs,use_reentrant=False)
        b.forward=types.MethodType(wrapped,b)
    try:
        emb=torch.load('artifacts/broad_coverage_20261030/embeddings/city_walk.pt',weights_only=True).cuda()
        for c in range(2):generate_chunk(engine,emb,0,c)
        x=noise(0,2,0);eps=noise(0,2,1);t=torch.full((3,),engine.steps[1],device='cuda',dtype=torch.float32)
        def next_input(clean):return engine.wrapper.scheduler.add_noise(clean.flatten(0,1),eps.flatten(0,1),t).unflatten(0,(1,3))
        rt.sparse=False;teacher1=engine.forward(x,emb,2,0);teacher2=engine.forward(next_input(teacher1),emb,2,1);rt.sparse=True
        hard1=engine.forward(x,emb,2,0);hard2=engine.forward(next_input(hard1),emb,2,1)
        commits=list(rt.commits);rt.training_event=True;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        first=Engine.forward.__wrapped__(engine,x,emb,2,0);first.retain_grad()
        second=Engine.forward.__wrapped__(engine,next_input(first),emb,2,1)
        assert torch.equal(first.detach(),hard1) and torch.equal(second.detach(),hard2)
        loss=(second.float()-teacher2.float()).square().mean()/teacher2.float().square().mean().clamp_min(1e-8)
        loss.backward();torch.cuda.synchronize()
        assert first.grad is not None and torch.isfinite(first.grad).all() and first.grad.norm()>0
        norms={n:float(p.grad.norm()) for n,p in rt.model.named_parameters()}
        assert all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in rt.model.parameters())
        assert all(p.grad is None for p in engine.wrapper.parameters()) and rt.commits==commits
        result=dict(passed=True,loss=float(loss.detach()),first_step_output_gradient_norm=float(first.grad.norm()),all_parameter_gradient_norms=norms,forward_backward_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),hard_forward_exact=True,scope='actual gradient through scheduler and both denoising forwards; fixed history, no cross-chunk BPTT; initialization only, not trained-video evidence')
        atomic(ROOT+'twostep_backward_checks.json',result);print(json.dumps({k:v for k,v in result.items() if k!='all_parameter_gradient_norms'}),flush=True)
    finally:rt.training_event=False;rt.close()
if __name__=='__main__':main()
