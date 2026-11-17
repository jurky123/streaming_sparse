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
    ap=argparse.ArgumentParser();ap.add_argument("--kind",choices=["a","co","both"],default="both");ap.add_argument("--vector-warps",type=int,choices=[1,2,4,8,16],default=4);ap.add_argument("--vector-pack",action="store_true");ap.add_argument("--vector-page32",action="store_true");ap.add_argument("--vector-maxnreg",type=int,default=0);ap.add_argument("--vector-test",action="store_true");ap.add_argument("--vector-width",type=int,choices=[4,8,16],default=16);ap.add_argument("--lazy",action="store_true");ap.add_argument("--split",action="store_true");ap.add_argument("--pages",type=int,choices=[216,288,360,432]);args=ap.parse_args()
    rr.LAYOUT_KIND=args.kind;rr.VECTOR_WARPS=args.vector_warps
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'artifacts/video_runtime_v1/reader_layout_v1'
    if args.split:root=ROOT/'artifacts/video_runtime_v1/reader_split_v1'
    if args.lazy:root=ROOT/'artifacts/video_runtime_v1/lazy_fallback_v1'
    if args.vector_test:root=ROOT/'artifacts/video_runtime_v1/reader_vector_v1'
    root=root/('kind_'+args.kind+(f'_p{args.pages}' if args.pages else '')+(f'_vw{args.vector_warps}' if args.vector_warps!=4 else '')+('_split' if args.split and args.lazy else '')+(f'_vd{args.vector_width}' if args.vector_test else '')+(f'_reg{args.vector_maxnreg}' if args.vector_maxnreg else '')+('_page32' if args.vector_page32 else '')+('_pack' if args.vector_pack else ''));root.mkdir(parents=True,exist_ok=True)
    _,_,phi,_,identity=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt',reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    saved=torch.load(ROOT/'artifacts/video_runtime_v1/hotspot_round9/e2e_risk_capture/route_entry.pt',weights_only=True)
    q=saved['q'].cuda();s=Summary(*(saved['summary'][n].cuda() for n in ('keys','mean','detail','stats')));en={k:v.cuda() for k,v in saved['en'].items()}
    if args.pages:
        s=Summary(*(getattr(s,n)[:,:args.pages].contiguous() for n in ('keys','mean','detail','stats')));en={k:v[:,:args.pages].contiguous() for k,v in en.items()}
    own=(torch.arange(72,device='cuda')+s.mean.shape[1]-72)[None].expand(12,-1).contiguous() if args.pages else saved['own'].cuda()
    probes=q.index_select(2,saved['offsets'].cuda());h,t,m,d=probes.shape;p=s.mean.shape[1]
    z=bf16_inputs_bmm(probes.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,m,p,4)/d**.5
    risk=risk_from_logits(probes,z,s,saved['theta'].cuda());mask=RouteOps.select(risk,own,Config(),check_valid=False)['mask'];r,a=candidate_pool(risk,mask,own)
    u=phi.query(probes);lm=z.double().logsumexp(-1);lm=lm-lm.logsumexp(-1,keepdim=True)
    rr.WARPS=4;rr.SHARD=128;rr.AGG32=True;rr.PACKED=True;rr.MOMENTS=True;rr.GROUP=1;rr.FMA=False;rr.PAGE32=False;rr.reader_kernel=reader_kernel;scores_moments.BLOCKED=True;scores_moments.LAZY=False
    rr.LAYOUT=True;rr.SPLIT=True;rr.VECTOR_WIDTH=16;rr.VECTOR_WARPS=4;rr.VECTOR_MAXNREG=0;rr.VECTOR_PACK=False
    mask=mask.reshape(h*t,p);r=r.reshape(h*t,-1);a=a.reshape(h*t,-1);nr=r.shape[-1];na=a.shape[-1]
    mapping=torch.full_like(mask,-1,dtype=torch.int32)
    mapping=mapping.scatter(1,r,torch.arange(nr,device='cuda',dtype=torch.int32)[None].expand(h*t,-1))
    mapping=mapping.scatter(1,a,torch.arange(nr,nr+na,device='cuda',dtype=torch.int32)[None].expand(h*t,-1))
    results=[]
    for flag in [False,True]:
        rr.PAGE32=flag
        results.append(rr.reader_reduce(u,en['co'],en['A'],en['mean'],en['detail'],z,lm,mask,mapping,nr+na,128,4,True,1,False))
    out={}
    for name,x,y in zip(['state','logs','candidates'],results[0],results[1]):
        finite=torch.isfinite(x)&torch.isfinite(y)
        out[name]=dict(equal=torch.equal(x,y),max_abs=float((x[finite]-y[finite]).abs().max()),different=int((x!=y).sum()))
    dest=ROOT/'artifacts/video_runtime_v1/reader_page32_v1/numerical_components.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
