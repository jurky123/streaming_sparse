"""Current captured Risk contract; preserve the original projection BMM."""
import argparse,json
from pathlib import Path
import torch
from .weights import ROOT
from .profile_route import capture
from .benchmark import measure
from .runtime import risk_from_logits
from . import risk_projected
from .tensor_bmm import bf16_inputs_bmm
from .scores import candidate_pool
from ..review_v1.ops import Summary,RouteOps,Config
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--per-probe",action="store_true");ap.add_argument("--sum-seq",action="store_true");ap.add_argument("--feature-seq",action="store_true");ap.add_argument("--sqrt-lib",action="store_true");ap.add_argument('--block',type=int,default=16);ap.add_argument('--warps',type=int,default=4);ap.add_argument('--pages',type=int,default=432);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;risk_projected.PAGE_BLOCK=args.block;risk_projected.WARPS=args.warps;risk_projected.SUM_SEQ=args.sum_seq;risk_projected.FEATURE_SEQ=args.feature_seq;risk_projected.SQRT_LIB=args.sqrt_lib;risk_projected.PER_PROBE=args.per_probe
    root=ROOT/f'artifacts/video_runtime_v1/risk_projected_v1/p{args.pages}_b{args.block}_w{args.warps}_s{int(args.sum_seq)}f{int(args.feature_seq)}l{int(args.sqrt_lib)}_per{int(args.per_probe)}';root.mkdir(parents=True,exist_ok=True)
    e=torch.load(ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True)
    q=e['q'].cuda().index_select(2,e['offsets'].cuda());s=Summary(*(e['summary'][n][:,:args.pages].cuda().contiguous() for n in ('keys','mean','detail','stats')));theta=e['theta'].cuda();h,t,m,d=q.shape;p=args.pages
    own=(torch.arange(t,device='cuda')+p-t)[None].expand(h,-1).contiguous()
    z=bf16_inputs_bmm(q.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,m,p,4)/(d**.5)
    funcs={n:torch.compile(lambda f=f:f(q,z,s,theta),fullgraph=True,options={'triton.cudagraphs':False}) for n,f in [('baseline',risk_from_logits),('projected',risk_projected.risk)]}
    graphs={};outputs={};rows={}
    for n,f in funcs.items():graphs[n],outputs[n]=capture(f)
    for _ in range(30):
        for g in graphs.values():g.replay()
    for n,g in graphs.items():rows[n]=measure(g.replay,50)
    x,y=outputs.values();mx=RouteOps.select(x,own,Config(),check_valid=False)['mask'];my=RouteOps.select(y,own,Config(),check_valid=False)['mask'];r,a=candidate_pool(x,mx,own);rr,aa=candidate_pool(y,my,own)
    result=dict(rows=rows,speedup=rows['baseline']['gpu_ms_median']/rows['projected']['gpu_ms_median'],bitwise=torch.equal(x,y),max_abs=float((x-y).abs().max()),nmse=float((x-y).square().sum()/x.square().sum()),mask_equal=torch.equal(mx,my),remove_equal=torch.equal(r,rr),add_equal=torch.equal(a,aa),mask_differences=int((mx!=my).sum()),kernel=risk_projected.KERNEL_STATS[p],shape=list(z.shape))
    for n,f in funcs.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:f();torch.cuda.synchronize()
        (root/(n+'_profile.txt')).write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
