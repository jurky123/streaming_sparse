"""A/B full route using deployed page count and native token-major Query strides."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .weights import ROOT
from .pipeline_graph import BatchGraphs
from .runtime import risk_from_logits
from .risk_fused import risk
from . import reader_reduce,scores_moments
from .reader_late import reader_kernel
from ..review_v1.ops import Config,Summary
from .benchmark import measure

@torch.no_grad()
def run(entry_path=None,warm_shapes=False,static_route=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    reader_reduce.WARPS=4;reader_reduce.SHARD=128;reader_reduce.AGG32=True;reader_reduce.MOMENTS=True;reader_reduce.PACKED=True;reader_reduce.reader_kernel=reader_kernel;scores_moments.BLOCKED=True
    ref,e,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    q=e['q'][:,:1].expand(-1,72,-1,-1)
    e['q']=q.permute(1,2,0,3).contiguous().permute(2,0,1,3)
    e['own']=e['own'][:,:1].expand(-1,72).contiguous()
    ix=torch.arange(432,device='cuda')%e['s'].mean.shape[1]
    e['s']=Summary(*(getattr(e['s'],n).index_select(1,ix) for n in ('keys','mean','detail','stats')))
    e['en']={k:v.index_select(1,ix) for k,v in e['en'].items()}
    if entry_path:
        saved=torch.load(entry_path,weights_only=True)
        e={k:v.cuda() for k,v in saved.items() if k not in ('summary','en')}
        e['s']=Summary(*(saved['summary'][n].cuda() for n in ('keys','mean','detail','stats')))
        e['en']={k:v.cuda() for k,v in saved['en'].items()}
    path=ROOT/('artifacts/video_runtime_v1/hotspot_round9/actual_static_shapes' if static_route else 'artifacts/video_runtime_v1/hotspot_round9/actual_warm_shapes' if warm_shapes else 'artifacts/video_runtime_v1/hotspot_round9/actual_entry' if entry_path else 'artifacts/video_runtime_v1/hotspot_round9/native_layout');path.mkdir(parents=True,exist_ok=True);rows={}
    for name,risk_fn in ([('after',risk)] if static_route else [('before',risk_from_logits),('after',risk)]):
        rt=BatchGraphs(risk_fn,ref.phi,Config(),compile_route=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True,static_route=static_route)
        if warm_shapes:
            for pages in (216,288,360):
                ep=dict(e);ep['s']=Summary(*(getattr(e['s'],n)[:,:pages].contiguous() for n in ('keys','mean','detail','stats')))
                ep['en']={k:v[:,:pages].contiguous() for k,v in e['en'].items()}
                ep['own']=(torch.arange(72,device='cuda')+pages-72)[None].expand(12,-1).contiguous()
                warm_graph,warm_out=capture(lambda:rt.compute(ep,True))
        fn=lambda:rt.compute(e,True);g,outputs=capture(fn);rows[name]=measure(g.replay,25)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:fn();torch.cuda.synchronize()
        (path/f'{name}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35));print(name,rows[name]['gpu_ms_median'],flush=True)
    (path/'result.json').write_text(json.dumps(dict(rows=rows,q_stride=list(e['q'].stride()),identity=identity,pages=432,scope=str(entry_path) if entry_path else 'Same captured event repeated; native Query strides, page fields extended modulo original410; layout/performance diagnostic only'),indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--entry');p.add_argument('--warm-shapes',action='store_true');p.add_argument('--static-route',action='store_true');a=p.parse_args();run(a.entry,a.warm_shapes,a.static_route)
