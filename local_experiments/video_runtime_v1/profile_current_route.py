"""Profile the deployed 72-tile route on one captured city_walk event."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .pipeline_graph import BatchGraphs
from .runtime import risk_from_logits
from .benchmark import measure
from .weights import ROOT
from . import reader_reduce
from .reader_late import reader_kernel
from ..review_v1.ops import Config

@torch.no_grad()
def run(blocked=False,fused_risk=False,output="artifacts/video_runtime_v1/hotspot_round7/current_route_profile"):
    from . import scores_moments
    scores_moments.BLOCKED=blocked
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    reader_reduce.WARPS=4;reader_reduce.SHARD=128;reader_reduce.AGG32=True;reader_reduce.MOMENTS=True;reader_reduce.PACKED=True;reader_reduce.reader_kernel=reader_kernel
    ref,e,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    e['q']=e['q'][:,:1].expand(-1,72,-1,-1).contiguous();e['own']=e['own'][:,:1].expand(-1,72).contiguous()
    risk_fn=risk_from_logits
    if fused_risk:
        from .risk_fused import risk
        risk_fn=risk
    rt=BatchGraphs(risk_fn,ref.phi,Config(),compile_route=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True)
    fn=lambda:rt.compute(e,True)
    graph,outputs=capture(fn);timing=measure(graph.replay,25)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn();torch.cuda.synchronize()
    out=ROOT/output;out.mkdir(parents=True,exist_ok=True)
    (out/'profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    prof.export_chrome_trace(str(out/'trace.json'))
    (out/'result.json').write_text(json.dumps(dict(timing=timing,identity=identity,shape=list(e['q'].shape),pages=e['s'].mean.shape[1],kernel_stats={str(k):v for k,v in reader_reduce.KERNEL_STATS.items()},scope='single captured city_walk event replicated across heads/tiles; excludes page staging'),indent=2));print(timing,flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--blocked',action='store_true');p.add_argument('--fused-risk',action='store_true');p.add_argument('--output',default='artifacts/video_runtime_v1/hotspot_round7/current_route_profile');a=p.parse_args();run(a.blocked,a.fused_risk,a.output)
