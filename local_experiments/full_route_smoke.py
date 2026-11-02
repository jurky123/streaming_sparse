"""Complete all-layer native-forward/gradient audit before training is launched."""
import argparse,json,time
from pathlib import Path
import torch
from .broad_common import Engine,noise
from .rollout_suite import generate_chunk
from .full_route_runtime import FullRouteRuntime
from .high_sparse_grad_smoke import timed
from .learned_cache.io import atomic
ROOT=Path('artifacts/full_route_grad_20261031');MODEL='artifacts/high_sparse_gain_20261031/model.pt'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--chunk',type=int,default=2);args=parser.parse_args()
    assert json.loads(Path('artifacts/route_graph_20261031/backward_checks.json').read_text())['passed']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False
    rt=FullRouteRuntime(engine,MODEL);emb=torch.load('artifacts/broad_coverage_20261030/embeddings/city_walk.pt',weights_only=True).cuda()
    try:
        for c in range(args.chunk):generate_chunk(engine,emb,0,c)
        x=noise(0,args.chunk,0);hard,_=timed(lambda:engine.forward(x,emb,args.chunk,0));n=list(rt.n);commits=list(rt.commits)
        rt.sparse=False;teacher,teacher_s=timed(lambda:engine.forward(x,emb,args.chunk,0));rt.sparse=True;rt.training_event=True
        memory=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        with torch.enable_grad():
            pred,forward_s=timed(lambda:Engine.forward.__wrapped__(engine,x,emb,args.chunk,0))
            fidelity=float((pred.detach().float()-hard.float()).square().sum()/hard.float().square().sum().clamp_min(1e-8));assert fidelity<1e-7,fidelity
            norm=teacher.float().square().mean().clamp_min(1e-8);loss=(pred.float()-teacher.float()).square().mean()/norm
            weights=list(rt.route_weights.values())
            grads,backward_s=timed(lambda:torch.autograd.grad(loss,weights,retain_graph=True))
            opposite=2*pred.detach().float()-teacher.float();reverse_loss=(pred.float()-opposite).square().mean()/norm
            reverse,_=timed(lambda:torch.autograd.grad(reverse_loss,weights))
        rows=[]
        for l,g,r in zip(rt.route_weights,grads,reverse):
            assert torch.isfinite(g).all() and g.norm()>0,(l,'missing gradient')
            cosine=float(torch.nn.functional.cosine_similarity(g.flatten(),r.flatten(),dim=0));assert cosine<-.995,(l,cosine)
            rows.append(dict(layer=l,grad_norm=float(g.norm()),reverse_error_cosine=cosine))
        assert all(p.grad is None for p in engine.wrapper.parameters());assert rt.n==n and rt.commits==commits
        result=dict(passed=True,layers=rows,hard_forward_bit_exact=torch.equal(pred.detach(),hard),hard_forward_nmse=fidelity,loss=float(loss.detach()),forward_seconds=forward_s,backward_seconds=backward_s,teacher_seconds=teacher_s,base_allocated_bytes=memory,peak_allocated_bytes=torch.cuda.max_memory_allocated(),extra_peak_bytes=torch.cuda.max_memory_allocated()-memory,backbone_grads=0,runtime=rt.finish())
        atomic(ROOT/('smoke.json' if args.chunk==2 else f'smoke_chunk{args.chunk}.json'),result);print(json.dumps(result),flush=True)
    finally:rt.training_event=False;rt.close()
if __name__=='__main__':main()
