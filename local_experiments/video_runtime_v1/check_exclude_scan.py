"""Full-shape same-input layout experiment, including packing and action scoring."""
import json,argparse
from pathlib import Path
import torch
from .weights import ROOT,load
from .profile_route import capture
from .benchmark import measure
from . import reader_reduce as rr,scores_moments
from .reader_late import reader_kernel
from .runtime import risk_from_logits
from .tensor_bmm import bf16_inputs_bmm
from .scores import candidate_pool
from ..review_v1.ops import Summary,RouteOps,Config
from ..learned_cache.io import atomic

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--batch-group",type=int,default=1);ap.add_argument("--batch-warps",type=int,default=4);ap.add_argument("--precompute",action="store_true");ap.add_argument("--shared-first",action="store_true");ap.add_argument("--score-block",type=int,default=32);ap.add_argument("--score-warps",type=int,default=4);ap.add_argument("--kind",choices=["a","co","both"],default="both");ap.add_argument("--vector-warps",type=int,choices=[1,2,4,8,16],default=4);ap.add_argument("--vector-pack",action="store_true");ap.add_argument("--vector-page32",action="store_true");ap.add_argument("--vector-maxnreg",type=int,default=0);ap.add_argument("--vector-test",action="store_true");ap.add_argument("--vector-width",type=int,choices=[4,8,16],default=16);ap.add_argument("--lazy",action="store_true");ap.add_argument("--split",action="store_true");ap.add_argument("--pages",type=int,choices=[216,288,360,432]);args=ap.parse_args()
    rr.LAYOUT_KIND=args.kind;rr.VECTOR_WARPS=args.vector_warps
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'artifacts/video_runtime_v1/reader_layout_v1'
    if args.split:root=ROOT/'artifacts/video_runtime_v1/reader_split_v1'
    if args.lazy:root=ROOT/'artifacts/video_runtime_v1/lazy_fallback_v1'
    if args.vector_test:root=ROOT/'artifacts/video_runtime_v1/reader_vector_v1'
    root=ROOT/'artifacts/video_runtime_v1/exclude_scan_v1'
    root=root/('kind_'+args.kind+(f'_p{args.pages}' if args.pages else '')+(f'_vw{args.vector_warps}' if args.vector_warps!=4 else '')+('_split' if args.split and args.lazy else '')+(f'_vd{args.vector_width}' if args.vector_test else '')+(f'_reg{args.vector_maxnreg}' if args.vector_maxnreg else '')+('_page32' if args.vector_page32 else '')+('_pack' if args.vector_pack else '')+f'_b{args.score_block}_w{args.score_warps}'+('_shared' if args.shared_first else '')+('_precompute' if args.precompute else '')+f'_g{args.batch_group}_bw{args.batch_warps}');root.mkdir(parents=True,exist_ok=True)
    _,_,phi,_,identity=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt',reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    saved=torch.load(ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True)
    q=saved['q'].cuda();s=Summary(*(saved['summary'][n].cuda() for n in ('keys','mean','detail','stats')));en={k:v.cuda() for k,v in saved['en'].items()}
    if args.pages:
        s=Summary(*(getattr(s,n)[:,:args.pages].contiguous() for n in ('keys','mean','detail','stats')));en={k:v[:,:args.pages].contiguous() for k,v in en.items()}
    own=(torch.arange(72,device='cuda')+s.mean.shape[1]-72)[None].expand(12,-1).contiguous() if args.pages else saved['own'].cuda()
    probes=q.index_select(2,saved['offsets'].cuda());h,t,m,d=probes.shape;p=s.mean.shape[1]
    z=bf16_inputs_bmm(probes.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,m,p,4)/d**.5
    risk=risk_from_logits(probes,z,s,saved['theta'].cuda());mask=RouteOps.select(risk,own,Config(),check_valid=False)['mask'];r,a=candidate_pool(risk,mask,own)
    u=phi.query(probes);lm=z.double().logsumexp(-1);lm=lm-lm.logsumexp(-1,keepdim=True)
    rr.WARPS=4;rr.SHARD=128;rr.AGG32=True;rr.PACKED=True;rr.MOMENTS=True;rr.GROUP=1;rr.FMA=False;rr.PAGE32=False;rr.reader_kernel=reader_kernel;scores_moments.BLOCKED=True;scores_moments.LAZY=False
    fn=torch.compile(lambda:rr.compact_scores(u,en,z,lm,mask,r,a),fullgraph=True,options={'triton.cudagraphs':False})
    rows={};graphs={};ref=None
    for name,flag in [('baseline',False),('layout',True)]:
        rr.LAYOUT=True if args.split or args.lazy or args.vector_test else flag;rr.SPLIT=flag if args.split else False;scores_moments.LAZY=flag if args.lazy else False;
        if args.vector_test:
            rr.VECTOR_PACK=args.vector_pack and flag;rr.PAGE32=args.vector_page32 and flag;rr.VECTOR_MAXNREG=args.vector_maxnreg if flag else 0;rr.SPLIT=True;rr.VECTOR_WIDTH=args.vector_width if flag else 16;rr.VECTOR_WARPS=args.vector_warps if flag else 4
        rr.LAYOUT=True;rr.LAYOUT_KIND="a";rr.SPLIT=True;scores_moments.FUSED_DEN=flag and not args.precompute;scores_moments.PRECOMPUTE_DEN=flag and args.precompute
        from . import scores_den
        scores_den.SHARED_FIRST=args.shared_first;scores_den.BLOCK=args.score_block;scores_den.WARPS=args.score_warps
        from . import scores,exclude_scan
        scores_moments.FUSED_DEN=False;scores_moments.PRECOMPUTE_DEN=False;scores.GROUP_BMM=False;scores.FUSED_EXCLUDE=False;scores.SCAN_EXCLUDE=flag
        exclude_scan.GROUP=args.batch_group;exclude_scan.WARPS=args.batch_warps
        g,x=capture(fn);graphs[name]=g
        if ref is None:ref=x.clone()
        delta=x-ref;rows[name]=dict(timing=measure(g.replay,25),bitwise_equal=torch.equal(x,ref),max_abs=float(delta.abs().max()),nmse=float(delta.square().sum()/ref.square().sum()),actions_equal=bool(torch.equal(x.flatten(1).argmax(1),ref.flatten(1).argmax(1))),noop_equal=bool(torch.equal(x.flatten(1).max(1).values>0,ref.flatten(1).max(1).values>0)),kernel=rr.KERNEL_STATS[(t,p)])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
        (root/f'{name}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
        print(name,rows[name],flush=True)
    for _ in range(25):
        for g in graphs.values():g.replay()
    for name,g in graphs.items():rows[name]['steady']=measure(g.replay,50)
    result=dict(lazy=args.lazy,vector_warps=args.vector_warps,split=args.split,layout_kind=args.kind,rows=rows,speedup=rows['baseline']['steady']['gpu_ms_median']/rows['layout']['steady']['gpu_ms_median'],shape=[h,t,m,p,d],identity=identity,theory=dict(page_coeff_bytes=h*p*(85+32)*4,scalar_A_stride_before=32,scalar_A_stride_after=1,beta_stride_before=85,beta_stride_after=1,response_coeff_row_before=85,response_coeff_row_after=80,precision='unchanged FP32 neural arithmetic / FP64 cached projections and scoring'),scope='FP64 small grouped BMM on current fast attention baseline input; lazy, PAGE32 and denominator fusion off',score_kernel={str(k):v for k,v in exclude_scan.KERNEL_STATS.items()})
    atomic(root/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
