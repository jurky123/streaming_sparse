"""Prove final-x0 loss reaches routing coefficients through a frozen suffix."""
import json,time
from pathlib import Path
import torch
from .broad_common import Engine,noise
from .rollout_suite import generate_chunk
from .high_sparse_grad_runtime import GradRuntime
from .learned_cache.io import atomic
ROOT=Path('artifacts/high_sparse_grad_20261031');MODEL='artifacts/high_sparse_gain_20261031/model.pt'
def timed(fn):
    torch.cuda.synchronize();t=time.perf_counter();x=fn();torch.cuda.synchronize();return x,time.perf_counter()-t

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False
    rt=GradRuntime(engine,MODEL);emb=torch.load('artifacts/broad_coverage_20261030/embeddings/city_walk.pt',weights_only=True).cuda()
    try:
        for c in range(2):generate_chunk(engine,emb,0,c)
        x=noise(0,2,0);hard,_=timed(lambda:engine.forward(x,emb,2,0))
        rt.sparse=False;teacher,teacher_s=timed(lambda:engine.forward(x,emb,2,0));rt.sparse=True
        before=rt.weight.detach().clone();opt=torch.optim.Adam([rt.weight],lr=.01)
        torch.cuda.reset_peak_memory_stats();base_memory=torch.cuda.memory_allocated();rt.training_event=True
        with torch.enable_grad():
            pred,fwd_s=timed(lambda:Engine.forward.__wrapped__(engine,x,emb,2,0))
            assert pred.requires_grad and pred.grad_fn is not None
            fidelity=float((pred.detach().float()-hard.float()).square().sum()/hard.float().square().sum().clamp_min(1e-8));assert fidelity<1e-7,fidelity
            norm=teacher.float().square().mean().clamp_min(1e-8)
            objective=(pred.float()-teacher.float()).square().mean()/norm
            g=torch.autograd.grad(objective,rt.weight,retain_graph=True)[0]
            opposite=2*pred.detach().float()-teacher.float()
            reversed_loss=(pred.float()-opposite).square().mean()/norm
            reverse=torch.autograd.grad(reversed_loss,rt.weight,retain_graph=True)[0]
            assert torch.isfinite(g).all() and g.norm()>0
            cosine=float(torch.nn.functional.cosine_similarity(g.flatten(),reverse.flatten(),dim=0));assert cosine<-.999,cosine
            opt.zero_grad();rt.last_margin.grad=None
            _,back_s=timed(objective.backward)
            assert rt.last_margin.grad is not None and rt.last_margin.grad.norm()>0
            assert all(p.grad is None for p in engine.wrapper.parameters())
            grad_norm=float(rt.weight.grad.norm());opt.step()
        change=float((rt.weight.detach()-before).norm());assert change>0
        rt.training_event=False;rt.clear_proxy();updated,_=timed(lambda:engine.forward(x,emb,2,0))
        updated_loss=float((updated.float()-teacher.float()).square().mean()/norm)
        result=dict(passed=True,final_x0_requires_grad=True,route_grad_norm=grad_norm,opposite_output_error_gradient_cosine=cosine,route_parameter_update_norm=change,backbone_parameter_grads=0,native_branch_checks=rt.native_checks,hard_forward_nmse=fidelity,hard_forward_bit_exact=torch.equal(pred.detach(),hard),before_loss=float(objective.detach()),after_hard_loss=updated_loss,student_forward_seconds=fwd_s,backward_seconds=back_s,teacher_seconds=teacher_s,base_allocated_bytes=base_memory,peak_allocated_bytes=torch.cuda.max_memory_allocated(),persistent_bytes=rt.persistent_bytes,scope='STE surrogate gradient, not derivative of discrete hard top-k; final output loss only; layer29 only; first-use timings')
        atomic(ROOT/'smoke.json',result);print(json.dumps(result),flush=True)
    finally:rt.close()
if __name__=='__main__':main()
