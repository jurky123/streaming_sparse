"""Measure the full exact+tail path with and without materialized tail outputs."""
import argparse
import json
import statistics
import torch
from .weights import ROOT
from .attention import Workspace
from .profile_route import capture
from .legacy_components import attention
from ..review_v1.ops import Summary

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--direct-output',action='store_true')
    parser.add_argument('--strided-tail',action='store_true')
    parser.add_argument('--fused-exact-tail',action='store_true')
    args=parser.parse_args()
    direct=args.direct_output
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'tail_liveness_v1/input.pt',weights_only=True)
    q,k,v=(data[key].cuda().contiguous() for key in ('q','k','v'))
    basis=data['basis'].cuda()
    rows=[]
    for p in (216,288,360,432):
        ids=(data['ids'][...,:p//10]%p).cuda().contiguous()
        summary=Summary(*(data['summary'][key][:,:p].cuda().contiguous() for key in ('keys','mean','detail','stats')))
        if args.strided_tail or args.fused_exact_tail:
            summary=Summary(*(data['summary'][key].cuda().contiguous()[:,:p] for key in ('keys','mean','detail','stats')))
        variants=[]
        for merge in (False,True):
            ws=Workspace();ws.tail_merge=direct or args.strided_tail or args.fused_exact_tail or merge
            ws.strided_tail=args.fused_exact_tail or (args.strided_tail and merge)
            ws.fused_exact_tail=args.fused_exact_tail and merge
            def run():
                out=attention(q,k[:,:p*64],v[:,:p*64],summary,basis,ids,ws)
                if direct and not merge:
                    result=ws.tensor('result',q.shape,q.device,q.dtype)
                    return result.copy_(out)
                return out
            graph,out=capture(run)
            variants.append((graph,out,ws))
        times=[[],[]]
        for rep in range(26):
            for i in (rep%2,1-rep%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(10):variants[i][0].replay()
                b.record();b.synchronize()
                if rep>=2:times[i].append(a.elapsed_time(b)/10)
        a,b=[x[1] for x in variants]
        ms=list(map(statistics.median,times))
        row=dict(pages=p,ms=ms,speedup=ms[0]/ms[1],bitwise=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),max_abs=(a.float()-b.float()).abs().max().item())
        print(row,flush=True);rows.append(dict(**row,samples_ms=times))
    path=root/('fused_exact_tail_v1' if args.fused_exact_tail else 'strided_tail_v1' if args.strided_tail else 'direct_output_v1' if direct else 'tail_merge_v1');path.mkdir(exist_ok=True)
    (path/'module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
