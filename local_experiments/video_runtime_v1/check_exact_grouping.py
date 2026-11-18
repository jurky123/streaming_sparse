"""Check raw scores, order, grouped KV, and four concurrent complete grouping graphs."""
import argparse
import json
import statistics
import torch
from .group_batch import ParallelGrouping
from .group_exact_direction import scores
from .weights import ROOT
from ..review_v1.model import Layer
from ..review_v1.ops import Config

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',action='store_true')
    ap.add_argument('--large',action='store_true')
    ap.add_argument('--root',action='store_true')
    ap.add_argument('--features',action='store_true')
    ap.add_argument('--staged',action='store_true')
    ap.add_argument('--combined',action='store_true')
    ap.add_argument('--sort',action='store_true')
    ap.add_argument('--radix',action='store_true')
    args=ap.parse_args()
    if args.radix:args.sort=True
    if args.sort:args.combined=True
    if args.features or args.staged or args.combined:args.root=True
    if args.root:args.large=True
    raw=args.raw or args.large
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(703)
    score_errors=[]
    for n in ((32,64,128,256,512) if args.large else (32,64,128)):
        g=4608*12//n
        x=torch.randn(g,n,32,device='cuda')
        xc=x-x.mean(1,keepdim=True)
        axis=xc.square().mean(1).argmax(-1,keepdim=True)
        y=xc.gather(2,axis[:,None].expand(-1,n,-1))
        for _ in range(4):
            v=torch.bmm(xc.transpose(1,2),y).squeeze(-1)
            v=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            y=torch.bmm(xc,v.unsqueeze(-1))
        candidate=scores(x) if raw else scores(xc,axis)
        d=candidate-y.squeeze(-1)
        score_errors.append(dict(n=n,differ=int((d!=0).sum()),max_abs=float(d.abs().max())))
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'grouping_dataflow_v1/input.pt',weights_only=True)
    m=Layer({k:v.cuda() for k,v in data['model'].items()}).eval()
    k,v=(data[n].cuda() for n in ('k','v'))
    inputs=[(m,k.roll(37*i,1),v.roll(37*i,1)) for i in range(4)]
    methods=[];values=[];samples=[[],[],[]]
    modes=('old','direction','raw') if raw else ('old','norm','direction')
    if args.large:modes=('direction','raw','large')
    if args.root:modes=('raw','large','root')
    if args.features:modes=('large','root','features')
    if args.staged:modes=('large','root','staged')
    if args.combined:modes=('root','staged','combined')
    if args.sort:modes=('combined','sort')
    if args.radix:modes=('sort','radix')
    samples=[[] for _ in modes]
    for mode in modes:
        direction={'radix':'root','sort':'root','staged':'root','features':'root','combined':'root','direction':True,'norm':False,'old':False}.get(mode,mode)
        op=ParallelGrouping(Config(history_pages=360),4,exact_small=direction,fused_norm=mode!='old',
                            fused_features=mode in ('features','combined','sort','radix'),staged_softmax=mode in ('staged','combined','sort','radix'))
        for group in op.groupers:group.fused_sort='radix' if mode=='radix' else mode=='sort'
        values.append(op(inputs));methods.append(op)
    for rep in range(24):
        for j in range(len(modes)):
            i=(rep+j)%len(modes)
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(4):methods[i](inputs)
            b.record();b.synchronize()
            if rep>=4:samples[i].append(a.elapsed_time(b)/16)
    rows=[]
    for mode,outs,times in zip(modes,values,samples):
        equal=[all(torch.equal(x,y) for x,y in zip(ref,new)) for ref,new in zip(values[0],outs)]
        rows.append(dict(mode=mode,ms=statistics.median(times),equal=equal,samples_ms=times))
    print(score_errors,[(r['mode'],r['ms'],r['equal']) for r in rows],flush=True)
    path=root/('group_sort_v1' if args.sort else 'group_feature_bundle_v1' if args.combined else 'group_staged_v1' if args.staged else 'group_features_v1' if args.features else 'root_group_direction_v1' if args.root else 'large_group_direction_v1' if args.large else 'raw_group_direction_v1' if raw else 'group_gemv_order_v1');path.mkdir(exist_ok=True)
    (path/('radix_grouping.json' if args.radix else 'grouping.json' if args.staged or args.sort else 'module.json')).write_text(json.dumps(dict(score_errors=score_errors,rows=rows),indent=2))
if __name__=='__main__':main()
