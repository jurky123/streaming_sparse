"""Existing stability fixtures plus same-city compact scorer timing."""
import json
from pathlib import Path
import torch
from . import validate,reader_reduce as module
from .scores_action import fused_moment_scores
from .profile_route import inputs,capture
from .scores import candidate_pool
from ..review_v1.ops import RouteOps,Config
from .benchmark import measure

@torch.no_grad()
def run(full_reader=False,output="artifacts/video_runtime_v1/hotspot_round7",skip_fixtures=False):
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    validate.grouped_scores=fused_moment_scores
    if not skip_fixtures:validate.run(str(path/'moment_validation'))
    module.AGG32=True;module.WARPS=4;module.SHARD=128;module.GROUP=1;module.PACKED=True
    from .weights import ROOT
    checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt' if full_reader else None
    rt,e,identity=inputs(True,True,reader_checkpoint=checkpoint);out=rt.compute(e,True)
    z=out[1].index_select(2,e['offsets']);q=e['q'].index_select(2,e['offsets']);u=rt.phi.query(q)
    lm=z.double().logsumexp(-1);lm=lm-lm.logsumexp(-1,keepdim=True)
    mask=RouteOps.select(out[3],e['own'],Config(),check_valid=False)['mask'];r,a=candidate_pool(out[3],mask,e['own'])
    fn=torch.compile(lambda:module.compact_scores(u,e['en'],z,lm,mask,r,a),fullgraph=True,options={'triton.cudagraphs':False})
    go,x=capture(fn);module.MOMENTS=True;gn,y=capture(fn)
    result=dict(before=measure(go.replay,25),after=measure(gn.replay,25),max_abs=float((x-y).abs().max()),actions_equal=bool(torch.equal(x.flatten(1).argmax(-1),y.flatten(1).argmax(-1))),noop_equal=bool(torch.equal(x.flatten(1).max(-1).values>0,y.flatten(1).max(-1).values>0)),identity=identity)
    assert torch.allclose(x,y,atol=1e-12,rtol=1e-9)
    assert result['actions_equal'] and result['noop_equal']
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn();torch.cuda.synchronize()
    (path/'profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
    (path/'moments.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--full-reader',action='store_true');p.add_argument('--skip-fixtures',action='store_true');p.add_argument('--output',default='artifacts/video_runtime_v1/hotspot_round7');a=p.parse_args();run(a.full_reader,a.output,a.skip_fixtures)
