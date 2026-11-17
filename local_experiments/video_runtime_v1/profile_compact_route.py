"""Retune query batch after eliminating expanded page responses."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .pipeline_graph import BatchGraphs
from .runtime import risk_from_logits
from .benchmark import measure
from .weights import ROOT
from . import reader_reduce
from ..review_v1.ops import Config

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    reader_reduce.AGG32=True;reader_reduce.SHARD=128;reader_reduce.WARPS=4;reader_reduce.GROUP=1;reader_reduce.PACKED=True
    old,base,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    path=Path('artifacts/video_runtime_v1/hotspot_round6/compact_profile');path.mkdir(parents=True,exist_ok=True);rows=[]
    for t in (6,12,24,72):
        e=dict(base);e['q']=base['q'][:,:1].expand(-1,t,-1,-1).contiguous();e['own']=base['own'][:,:1].expand(-1,t).contiguous()
        rt=BatchGraphs(risk_from_logits,old.phi,Config(),compile_route=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True)
        fn=lambda:rt.compute(e,True);g,out=capture(fn);timing=measure(g.replay,25)
        row=dict(tiles=t,ms=timing['gpu_ms_median'],layer_ms=timing['gpu_ms_median']*72/t);rows.append(row);print(row,flush=True)
        if t in (6,24):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:fn();torch.cuda.synchronize()
            (path/f'route_t{t}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
        del g,out
        (path/'result.json').write_text(json.dumps(dict(rows=rows,identity=identity),indent=2))
if __name__=='__main__':run()
