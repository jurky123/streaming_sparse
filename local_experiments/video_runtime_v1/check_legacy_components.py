"""Isolated current-versus-previous kernels: speed, outputs, and CUDA profiles."""
import argparse,json,math
from pathlib import Path
import torch
from .weights import ROOT,load
from .profile_route import inputs,capture
from .benchmark import measure
from .attention import Workspace
from .stream_attention import mixed_attention,full_values
from .runtime import risk_from_logits
from . import legacy_components as old
from ..review_v1.ops import Summary,RouteOps,Config
from ..learned_cache.io import atomic

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['rope','risk','attention','summary'],required=True);ap.add_argument('--pages',type=int);a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'artifacts/video_runtime_v1/legacy_reuse'/(a.part+(str(a.pages) if a.pages else ''));out.mkdir(parents=True,exist_ok=True)
    funcs={};extra={}
    if a.part=='risk':
        saved=torch.load(ROOT/'artifacts/video_runtime_v1/hotspot_round9/e2e_risk_capture/route_entry.pt',weights_only=True)
        q=saved['q'].cuda();s=Summary(*(saved['summary'][n].cuda() for n in ('keys','mean','detail','stats')));theta=saved['theta'].cuda()
        probes=q.index_select(2,saved['offsets'].cuda());h,t,m,d=probes.shape;p=s.mean.shape[1]
        z=torch.bmm(probes.bfloat16().flatten(1,2),s.keys.flatten(1,2).transpose(1,2)).float().reshape(h,t,m,p,4)/math.sqrt(d)
        # Match current bmm's FP32 output (BF16 inputs, FP32 accumulation).
        from .tensor_bmm import bf16_inputs_bmm
        z=bf16_inputs_bmm(probes.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,m,p,4)/math.sqrt(d)
        current=torch.compile(risk_from_logits,fullgraph=True);funcs=dict(current=lambda:current(probes,z,s,theta),legacy=lambda:old.risk(probes,z,s,theta),legacy_compiled=lambda:compiled_old(probes,z,s,theta))
        compiled_old=torch.compile(old.risk,fullgraph=True)
        extra=dict(shape=list(z.shape),shared_qk=True,qk_gflops=2*h*t*m*p*4*d/1e9)
    elif a.part=='rope':
        from wan.modules.causal_model import causal_rope_apply
        from wan.modules.model import rope_params
        from ..review_v1.fused_rope import apply
        q=torch.randn(1,4608,12,128,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);grid=torch.tensor([[3,32,48]])
        freq=torch.cat([rope_params(1024,44),rope_params(1024,42),rope_params(1024,42)],1).cuda()
        funcs=dict(current=lambda:(causal_rope_apply(q,grid,freq,24).bfloat16().contiguous(),causal_rope_apply(k,grid,freq,24).bfloat16().contiguous()),legacy=lambda:apply(q,k,grid,freq,24))
        extra=dict(shape=list(q.shape),input_output_bytes=4*q.numel()*q.element_size(),minimum_fp64_flops=2*q.numel()*3)
    else:
        rt,e,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
        path=next((ROOT/'artifacts/full_coverage_v1/data/city_walk_s701').glob('c8_l15_s0_h0_t*.pt'));ev=torch.load(path,weights_only=True);kv=torch.load(path.parent/ev['kv_file'],weights_only=True)
        h=12;s=e['s'];p=s.mean.shape[1]
        rk=kv['k'].cuda()[None].expand(h,-1,-1,-1).contiguous();rv=kv['v'].cuda()[None].expand_as(rk).contiguous()
        if a.part=='attention':
            if a.pages:
                ix=torch.arange(a.pages,device='cuda')%p;s=Summary(*(getattr(s,name).index_select(1,ix) for name in ('keys','mean','detail','stats')));rk=rk.index_select(1,ix);rv=rv.index_select(1,ix);p=a.pages
            q=e['q'][:,:1].expand(-1,72,-1,-1).contiguous();n=4608;d=128
            nq=q.flatten(1,2).bfloat16().transpose(0,1).contiguous()[None]
            nk=rk.flatten(1,2).transpose(0,1).contiguous()[None];nv=rv.flatten(1,2).transpose(0,1).contiguous()[None]
            ids=torch.arange(int(p*.1),device='cuda',dtype=torch.int32)[None,None].expand(h,72,-1).contiguous();mask=torch.zeros(h,72,p,device='cuda',dtype=torch.bool).scatter(-1,ids.long(),True)
            values=full_values(s,e['basis']);w=Workspace();wold=Workspace()
            funcs=dict(current=lambda:mixed_attention(nq,nk,nv,s,values,ids,mask,w,tail_mode=4,warps=8),legacy=lambda:old.attention(nq,nk,nv,s,e['basis'],ids,wold),legacy_tf32=lambda:old.attention(nq,nk,nv,s,e['basis'],ids,wold,bf16_av=False))
            extra=dict(shape=list(nq.shape),pages=p,logical_rep_av_gflops=2*h*n*p*4*d/1e9,current_issued_tail_av_gflops=2*h*n*p*4*d*4/1e9,legacy_issued_tail_av_gflops=2*h*n*p*(d+3*8)*2/1e9,scope='Captured single head/tile repeated for cost isolation; no page-value reconstruction included in current')
        else:
            from ..review_v1.split_summary import build
            model,*_=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt',reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
            rk=rk[:,:72].contiguous();rv=rv[:,:72].contiguous()
            funcs=dict(current=lambda:build(model[15],rk,rv,True),legacy=lambda:build(model[15],rk,rv,True,three_term_features=True))
            extra=dict(shape=list(rk.shape),scope='Current noisy summary; all four output fields compared; legacy three-term BF16 feature projection versus current FP32 projection')
    rows={};reference=None;graphs={}
    def flatten(x):
        if isinstance(x,Summary):return [getattr(x,n).float().flatten() for n in ('keys','mean','detail','stats')]
        return [t.float().flatten() for t in (x if isinstance(x,tuple) else (x,))]
    for name,fn in funcs.items():
        graph,x=capture(fn);graphs[name]=graph;timing=measure(graph.replay,30);v=torch.cat(flatten(x))
        if reference is None:reference=v.clone()
        delta=v-reference;rows[name]=dict(timing=timing,bitwise_equal=torch.equal(v,reference),nmse=float(delta.square().sum()/reference.square().sum()),max_abs=float(delta.abs().max()),finite=bool(v.isfinite().all()))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            graph.replay();torch.cuda.synchronize()
        (out/f'{name}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
        print(name,rows[name],flush=True)
    for _ in range(20):
        for graph in graphs.values():graph.replay()
    torch.cuda.synchronize()
    for name,graph in graphs.items():rows[name]['steady_timing']=measure(graph.replay,60)
    for row in rows.values():row['speedup']=rows['current']['steady_timing']['gpu_ms_median']/row['steady_timing']['gpu_ms_median']
    atomic(out/'result.json',dict(part=a.part,rows=rows,extra=extra))
if __name__=='__main__':main()
