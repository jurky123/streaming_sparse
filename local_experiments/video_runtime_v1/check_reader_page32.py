"""Read the already-FP32-used page projections as FP32; include conversion cost."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .weights import ROOT
from .pipeline_graph import BatchGraphs
from .runtime import risk_from_logits
from .reader_late import reader_kernel
from . import reader_reduce as module,scores_moments
from .scores import candidate_pool
from ..review_v1.ops import Summary,Config,RouteOps
from .benchmark import measure

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    module.WARPS=4;module.SHARD=128;module.AGG32=True;module.MOMENTS=True;module.PACKED=True;module.reader_kernel=reader_kernel;scores_moments.BLOCKED=True
    ref,_,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    source=ROOT/'artifacts/video_runtime_v1/hotspot_round9/e2e_risk_capture/route_entry.pt';saved=torch.load(source,weights_only=True)
    e={k:v.cuda() for k,v in saved.items() if k not in ('en','summary')};e['s']=Summary(*(saved['summary'][n].cuda() for n in ('keys','mean','detail','stats')));e['en']={k:v.cuda() for k,v in saved['en'].items()}
    rt=BatchGraphs(risk_from_logits,ref.phi,Config(),compile_route=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True,static_route=True)
    out=rt.compute(e,True);z=out[1];q=e['q'].index_select(2,e['offsets']);u=ref.phi.query(q)
    lm=z.double().logsumexp(-1);lm=lm-lm.logsumexp(-1,keepdim=True);mask=RouteOps.select(out[3],e['own'],Config(),check_valid=False)['mask'];r,a=candidate_pool(out[3],mask,e['own'])
    fn=torch.compile(lambda:module.compact_scores(u,e['en'],z,lm,mask,r,a),fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    module.PAGE32=False;go,x=capture(fn);module.PAGE32=True;gn,y=capture(fn)
    result=dict(before=measure(go.replay,25),after=measure(gn.replay,25),bitwise_equal=bool(torch.equal(x,y)),max_abs=float((x-y).abs().max()),actions_equal=bool(torch.equal(x.flatten(1).argmax(-1),y.flatten(1).argmax(-1))),noop_equal=bool(torch.equal(x.flatten(1).max(-1).values>0,y.flatten(1).max(-1).values>0)),kernel_stats={str(k):v for k,v in module.KERNEL_STATS.items()},identity=identity,source=str(source),notes='Mean/detail are already cast to FP32 inside AGG32 Reader. Candidate moves that same cast before kernel and includes conversion in measured CUDA Graph.')
    path=ROOT/'artifacts/video_runtime_v1/hotspot_round10';path.mkdir(parents=True,exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:fn();torch.cuda.synchronize()
    (path/'page32_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
    (path/'page32.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    assert result['bitwise_equal'], 'Bitwise gate failed; diagnostics saved'
if __name__=='__main__':run()
