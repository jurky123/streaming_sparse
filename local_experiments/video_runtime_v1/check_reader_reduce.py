"""Same-city Reader/projection/scoring reference versus compact reduction."""
import json,math
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .page import read
from .scores import candidate_pool
from .scores_action import fused_action_scores
from .reader_reduce import compact_scores
from .benchmark import measure

@torch.no_grad()
def run(warps=8,event_path=None,full_reader=False,output=None,shard=128,aggregate_float=False,group=1,fma=False):
    from . import reader_reduce as module
    module.WARPS=warps;module.SHARD=shard;module.AGG32=aggregate_float;module.GROUP=group;module.FMA=fma
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from .weights import ROOT
    reader_path=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt' if full_reader else None
    rt,e,identity=inputs(True,True,event_path,reader_path);out=rt.compute(e,True)
    q=e['q'];h,t,n,d=q.shape;p=e['s'].mean.shape[1];b=h*t
    z=out[1].index_select(2,e['offsets']);probes=q.index_select(2,e['offsets'])
    from ..review_v1.ops import RouteOps,Config
    mask=RouteOps.select(out[3],e['own'],Config(),check_valid=False)['mask']
    r,a=candidate_pool(out[3],mask,e['own']);en=e['en'];zz=z.double();lm=zz.logsumexp(-1);lm=lm-lm.logsumexp(-1,keepdim=True)
    def old():
        rp,beta=read(probes,en['co'],en['A'],rt.phi)
        contrast=torch.einsum('htmpr,ar->htmpa',zz.softmax(-1),e['haar'])
        approx=en['mean'][:,None,None]+torch.einsum('htmpa,hpak->htmpk',contrast,en['detail'])
        center=approx-(lm.exp()[...,None]*approx).sum(-2,keepdim=True)
        return fused_action_scores(rp.reshape(b,16,p,16),beta.reshape(b,16,p),lm.reshape(b,16,p),center.reshape(b,16,p,16),mask.reshape(b,p),r.reshape(b,-1),a.reshape(b,-1))
    def new():return compact_scores(rt.phi.query(probes),en,z,lm,mask,r,a)
    old=torch.compile(old,fullgraph=True,options={'triton.cudagraphs':False});new=torch.compile(new,fullgraph=True,options={'triton.cudagraphs':False})
    go,x=capture(old);gn,y=capture(new)
    assert torch.isfinite(y).all()
    result=dict(kernel_stats={str(k):v for k,v in module.KERNEL_STATS.items()},max_abs=float((x-y).abs().max()),score_nmse=float((x-y).square().sum()/x.square().sum()),argmax_mismatches=int((x.flatten(1).argmax(1)!=y.flatten(1).argmax(1)).sum()),noop_mismatches=int(((x.flatten(1).max(1).values>0)!=(y.flatten(1).max(1).values>0)).sum()),before=measure(go.replay,25),after=measure(gn.replay,25),before_eager=measure(old,5),after_eager=measure(new,5),identity=identity)
    path=Path(output or 'artifacts/video_runtime_v1/hotspot_round6');path.mkdir(parents=True,exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:new();torch.cuda.synchronize()
    (path/f'reader_profile_w{warps}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=20))
    (path/f'reader_reduce_w{warps}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--warps',type=int,choices=[4,8,16],default=8);p.add_argument('--fma',action='store_true');p.add_argument('--group',type=int,choices=[1,2,4],default=1);p.add_argument('--aggregate-float',action='store_true');p.add_argument('--shard',type=int,choices=[32,64,128,256,512],default=128);p.add_argument('--event');p.add_argument('--full-reader',action='store_true');p.add_argument('--output');a=p.parse_args();run(a.warps,a.event,a.full_reader,a.output,a.shard,a.aggregate_float,a.group,a.fma)
