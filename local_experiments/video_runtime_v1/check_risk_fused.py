"""Same-input Risk and candidate-pool contract, timings and fresh profiles."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .weights import ROOT
from .runtime import risk_from_logits
from .risk_fused import risk
from .scores import candidate_pool
from ..review_v1.ops import RouteOps,Config
from .benchmark import measure

@torch.no_grad()
def run(pages=410):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rt,e,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt');out=rt.compute(e,True)
    expand=lambda x:x[:,:1].expand(-1,72,*x.shape[2:]).contiguous()
    z=expand(out[1].index_select(2,e['offsets']));q=expand(e['q'].index_select(2,e['offsets']));own=expand(e['own'])
    if pages!=z.shape[-2]:
        from ..review_v1.ops import Summary
        index=torch.arange(pages,device='cuda')%z.shape[-2]
        z=z.index_select(-2,index)
        e['s']=Summary(*(getattr(e['s'],n).index_select(1,index) for n in ('keys','mean','detail','stats')))
    before=torch.compile(lambda:risk_from_logits(q,z,e['s'],e['theta']),fullgraph=True,options={'triton.cudagraphs':False})
    after=torch.compile(lambda:risk(q,z,e['s'],e['theta']),fullgraph=True,options={'triton.cudagraphs':False})
    go,x=capture(before);gn,y=capture(after)
    mask=lambda s:RouteOps.select(s,own,Config(),check_valid=False)['mask']
    r,a=candidate_pool(x,mask(x),own);rr,aa=candidate_pool(y,mask(y),own)
    result=dict(before=measure(go.replay,25),after=measure(gn.replay,25),nmse=float((x-y).square().sum()/x.square().sum()),max_abs=float((x-y).abs().max()),selection_equal=bool(torch.equal(mask(x),mask(y))),remove_equal=bool(torch.equal(r,rr)),add_equal=bool(torch.equal(a,aa)),identity=identity,shape=list(z.shape))
    assert torch.allclose(x,y,rtol=1e-5,atol=1e-6)
    p=ROOT/'artifacts/video_runtime_v1/hotspot_round9'/f'pages{pages}';p.mkdir(parents=True,exist_ok=True)
    for name,fn in [('before',before),('after',after)]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:fn();torch.cuda.synchronize()
        (p/f'risk_{name}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
    (p/'risk.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--pages',type=int,default=432);run(p.parse_args().pages)
