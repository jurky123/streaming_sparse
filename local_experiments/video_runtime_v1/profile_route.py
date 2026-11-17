"""Single-city-event route/tail profiles and isolated CUDA-graph timing."""
import argparse,json
from pathlib import Path
import torch
from .weights import ROOT,load
from .runtime import risk_from_logits
from .pipeline_graph import BatchGraphs
from .page import encode
from .attention import compensate
from .benchmark import measure
from ..review_v1.ops import Summary,Config,haar
from ..learned_cache.io import atomic

@torch.no_grad()
def inputs(action_kernel=False,tensor_math=False,event_path=None,reader_checkpoint=None):
    model,reader,phi,B,identity=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt',reader_checkpoint=reader_checkpoint)
    path=Path(event_path) if event_path else next((ROOT/'artifacts/full_coverage_v1/data/city_walk_s701').glob('c8_l15_s0_h0_t*.pt'));
    import re
    layer=int(re.search(r'_l(\d+)_',path.name)[1]);ev=torch.load(path,weights_only=True);kv=torch.load(path.parent/ev['kv_file'],weights_only=True)
    h,t=12,6;s=Summary(*(kv['summary'][n].cuda().expand(h,-1,*kv['summary'][n].shape[2:]).contiguous() for n in ('keys','mean','detail','stats')))
    q=ev['q'].cuda().float()[None,None].expand(h,t,-1,-1).contiguous();co,A,mean=encode(s,reader,phi,B.double());basis=model[layer].basis
    en=dict(co=co,A=A,mean=mean,detail=torch.matmul(s.detail.double(),(basis.bfloat16().double().transpose(1,2)@B.double())[:,None]))
    e=dict(q=q,s=s,en=en,basis=basis,theta=model[layer].theta,own=torch.full((h,t),ev['own'],device='cuda',dtype=torch.long),offsets=torch.linspace(0,63,16,device='cuda').round().long(),haar=haar(q).double(),haar_float=haar(q))
    return BatchGraphs(risk_from_logits,phi,Config(),compile_route=True,action_kernel=action_kernel,tensor_math=tensor_math),e,identity


def capture(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    g.replay();torch.cuda.synchronize();return g,out

@torch.no_grad()
def run(a):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rt,e,identity=inputs(a.action_kernel);route=rt.compute(e,True)
    tail=torch.compile(compensate,fullgraph=True,options={'triton.cudagraphs':False}) if a.compile_tail else compensate
    funcs=dict(route=lambda:rt.compute(e,True),tail=lambda:tail(route[1],e['s'],e['basis'],route[2],e['haar_float']))
    rows={}
    for name,fn in funcs.items():
        graph,_=capture(fn);rows[name]=measure(graph.replay,15)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            fn();torch.cuda.synchronize()
        (out/f'{name}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
        prof.export_chrome_trace(str(out/f'{name}_trace.json'))
    h,t,n,d=e['q'].shape;p=e['s'].mean.shape[1]
    flops=dict(Q64_R4_QK=2*h*t*n*p*4*d,full_mean_AV=2*h*t*n*p*d,rank8_detail=2*h*t*n*p*3*8,reader_coeff=2*h*t*16*p*4*17,phi_page_query=2*h*t*16*p*8*4,center_projection=2*h*t*16*p*3*16)
    atomic(out/'result.json',dict(rows=rows,flops_per_6_tiles=flops,shape=[h,t,n,p,d],identity=identity));print(json.dumps(rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action-kernel',action='store_true');p.add_argument('--compile-tail',action='store_true');p.add_argument('--output',default='artifacts/video_runtime_v1/hotspot_round3/profile');run(p.parse_args())
