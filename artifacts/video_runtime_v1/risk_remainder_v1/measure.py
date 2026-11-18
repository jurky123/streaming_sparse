"""Benchmark specializing the final Risk tile while preserving its global page offsets."""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
from statistics import median
import tempfile

import torch
import triton
from local_experiments.video_runtime_v1.risk_bound import bound
from local_experiments.video_runtime_v1.risk_partial_mass import _partial, _finish
from local_experiments.video_runtime_v1.profile_route import capture


def generate(path, separate):
    src = inspect.getsource(_partial.fn)
    src = src.replace('def _partial(', 'def body(').replace('CHANNEL_MAJOR:tl.constexpr=False', 'WIDTH:tl.constexpr,CHANNEL_MAJOR:tl.constexpr=False')
    src = src.replace('tl.arange(0,4*BP)', 'tl.arange(0,4*WIDTH)').replace('tl.arange(0,BP)', 'tl.arange(0,WIDTH)')
    src = src.replace('(16,4*BP)', '(16,4*WIDTH)').replace('(16,BP,4)', '(16,WIDTH,4)')
    wrapper = """
@triton.jit
def specialized(Q,K,U,QN,ST,TH,C,VAL,NORM,T:tl.constexpr,P:tl.constexpr,BP:tl.constexpr,CHANNEL_MAJOR:tl.constexpr):
    if tl.program_id(1) == triton.cdiv(P,BP)-1:
        body(Q,K,U,QN,ST,TH,C,VAL,NORM,T,P,BP,triton.next_power_of_2(P%BP),CHANNEL_MAJOR)
    else:
        body(Q,K,U,QN,ST,TH,C,VAL,NORM,T,P,BP,BP,CHANNEL_MAJOR)
"""
    if separate:
        src=src.replace('WIDTH:tl.constexpr,CHANNEL_MAJOR','OFFSET:tl.constexpr,WIDTH:tl.constexpr,CHANNEL_MAJOR').replace('block=tl.program_id(1)','block=tl.program_id(1)+OFFSET')
        wrapper=''
    path.write_text('import triton\nimport triton.language as tl\nfrom triton.language.extra.cuda import libdevice\n'+src+wrapper)
    spec=importlib.util.spec_from_file_location('partial_remainder',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.body if separate else module.specialized


def measure(path, candidate, separate, tuning=None):
    event=torch.load(path,weights_only=True)
    q,k,stats,theta=(event[key].cuda().contiguous() for key in ('q','keys','stats','theta'))
    h,t=q.shape[:2]
    pages=stats.shape[1]
    width=128
    blocks=triton.cdiv(pages,width)
    u=q.float().square().reshape(h,t,16,8,16).sum(-1)/128
    qn=q.float().norm(dim=-1)/(128**.5)
    c=bound(stats).contiguous()
    def run(kernel, config):
        width,warps,stages,registers=config
        blocks=triton.cdiv(pages,width)
        options=dict(num_warps=warps,num_stages=stages,enable_fp_fusion=False)
        if registers:options["maxnreg"]=registers
        st=stats.flatten(1,2).transpose(1,2).contiguous()
        v=u.new_empty((h*t,blocks,16,width))
        n=u.new_empty((h*t,blocks,16,2))
        out=u.new_empty((h,t,pages))
        args=(q,k,u,qn,st,theta,c,v,n,t,pages,width)
        if separate and kernel is candidate:
            kernel[(h*t,blocks-1)](*args,0,width,True,num_warps=4,enable_fp_fusion=False)
            kernel[(h*t,1)](*args,blocks-1,triton.next_power_of_2(pages%width),True,num_warps=4,enable_fp_fusion=False)
        else:
            kernel[(h*t,blocks)](*args,True,**options)
        _finish[(h*t,blocks)](v,n,out,pages,width,triton.next_power_of_2(blocks),num_warps=4,enable_fp_fusion=False)
        return out
    configs=[(128,4,3,0)]*2
    kernels=[_partial,candidate]
    if tuning:
        configs=tuning
        kernels=[_partial]*len(configs)
    cases=[capture(lambda: run(kernel,config)) for kernel,config in zip(kernels,configs)]
    samples=[[] for _ in cases]
    for rep in range(24):
        for j in range(len(cases)):
            idx=(rep+j)%len(cases)
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(16):
                cases[idx][0].replay()
            b.record()
            b.synchronize()
            if rep>=4:samples[idx].append(a.elapsed_time(b)/16)
    x,y=(case[1] for case in cases[:2])
    own=event['own'].cuda()
    def ids(z):return z.scatter(-1,own[...,None],float('inf')).topk(pages//10).indices
    if tuning:
        return [dict(config=config,ms=median(times),speedup=median(a/b for a,b in zip(samples[0],times)),max_abs=(x-z).abs().max().item(),different_ids=(ids(x)!=ids(z)).sum().item()) for config,times,(_,z) in zip(configs,samples,cases)]
    return dict(event=path.stem,ms=list(map(median,samples)),speedup=median(a/b for a,b in zip(*samples)),max_abs=(x-y).abs().max().item(),different_ids=(ids(x)!=ids(y)).sum().item(),samples=samples)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--separate',action='store_true')
    parser.add_argument('--tune',action='store_true')
    parser.add_argument('--stages',action='store_true')
    args=parser.parse_args()
    tuning=([(128,4,3,0),(128,8,2,0),(128,16,2,0),(128,4,2,128),(128,4,2,96),(64,4,2,0),(64,8,2,0),(32,4,2,0)] if args.tune else None)
    if args.stages:
        tuning=[(128,4,3,0),(128,4,1,0),(128,4,2,0),(128,4,4,0),(128,4,5,0),(128,8,3,0),(64,4,3,0),(64,8,3,0),(32,4,3,0)]
    with tempfile.TemporaryDirectory() as tmp:
        result=measure(args.input,generate(Path(tmp)/'partial_remainder.py',args.separate),args.separate,tuning)
    Path(__file__).with_name(args.input.stem+('_stages' if args.stages else '_tune' if tuning else '_separate' if args.separate else '')+'.json').write_text(json.dumps(result,indent=2))
