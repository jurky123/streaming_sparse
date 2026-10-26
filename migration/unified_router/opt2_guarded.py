"""Fast FP32-equivalent score plus guarded per-query action evaluation."""
import torch
import triton as tr
from .opt2.network import load_optimized
from .opt2_precision import fma_score
from .opt2.features import _statistics
from .opt2_guarded_kernels import _guard_actions,_reference_fallback
from .fast_graph import GraphCache

def guarded_finish(q,sketch,score,mask,rem,add,beta,step,value,head,return_flags=False):
    h,n,b=score.shape
    stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
    out=torch.empty_like(mask);flags=torch.empty((h*n,),device=q.device,dtype=torch.bool)
    _statistics[(h*n,)](score,value,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),num_warps=4,enable_fp_fusion=False)
    _guard_actions[(h*n,)](flags,stats,score,value,mask,sketch.costs,sketch.origins,rem,add,
        head.w1,head.b1,head.w2,head.b2,head.mean,head.std,step,out,n,b,tr.next_power_of_2(b),
        sketch.history_tokens,sketch.current_tokens,beta,num_warps=4,enable_fp_fusion=False)
    _reference_fallback[(h*n,)](flags,score,value,mask,sketch.costs,sketch.origins,rem,add,
        head.w1,head.b1,head.w2,head.b2,head.mean,head.std,step,out,n,b,tr.next_power_of_2(b),
        sketch.history_tokens,sketch.current_tokens,beta,num_warps=8,enable_fp_fusion=False)
    return (out,flags)if return_flags else out

def load_guarded(directory,training_seed=0,**options):
    net,m=load_optimized(directory,training_seed,**options)
    net.fast_route.score=fma_score;net.fast_route.finish=guarded_finish
    net.graph_route=GraphCache(net.fast_route,net.route_batch)
    return net,m

def main():
    from . import fast_bench
    fast_bench.load_fast=load_guarded;fast_bench.main()

if __name__=='__main__':main()
