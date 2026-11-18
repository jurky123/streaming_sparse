"""Test removing the per-page LUT load from the exact-attention loop."""
import argparse
import json
import importlib.util
import tempfile
from pathlib import Path
import statistics
import torch
from .weights import ROOT
from .profile_route import capture
from ..gap_exact_round_kernel import _attn_fwd

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefetch',type=int,default=0)
    parser.add_argument('--query-group',type=int,default=0)
    parser.add_argument('--unroll',type=int)
    parser.add_argument('--loop-stages',type=int)
    parser.add_argument('--stages',type=int)
    parser.add_argument('--max-reg',type=int)
    parser.add_argument('--channels',type=int,choices=(32,64))
    args=parser.parse_args()
    candidate=_attn_fwd
    if args.unroll or args.loop_stages:
        source=Path(_attn_fwd.fn.__code__.co_filename).read_text()
        option=f'loop_unroll_factor={args.unroll}' if args.unroll else f'num_stages={args.loop_stages}'
        folder=tempfile.TemporaryDirectory()
        path=Path(folder.name)/'exact_loop_candidate.py'
        path.write_text(source.replace('tl.range(topk):',f'tl.range(topk,{option}):'))
        spec=importlib.util.spec_from_file_location('exact_loop_candidate',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate=module._attn_fwd
    if args.channels:
        from .exact_channel_split import _attn_fwd as candidate
    distance=args.prefetch
    group=args.query_group
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'tail_liveness_v1/input.pt',weights_only=True)
    q,k,v=(data[key].cuda().contiguous() for key in ('q','k','v'))
    results=[]
    for pages in (216,288,360,432):
        count=int(.1*pages)
        ids=(data['ids'][...,:count].cuda()%pages).contiguous()[None]
        graphs=[];outputs=[];resources=[]
        for cache in (False,True):
            out=torch.empty_like(q,dtype=torch.float32)
            lse=q.new_empty(12,4608,dtype=torch.float32)
            def call():
                grid=(72,12,128//args.channels) if args.channels and cache else (72,12)
                return (candidate if cache else _attn_fwd)[grid](q,k,v,128**-.5,count,ids,lse,out,12,4608,pages*64,
                                         72,128,64,64,HIGH_PRECISION_AV=True,QUERY_GROUP=group if cache else 0,CACHE_LUT=cache and not distance and not group and not args.unroll and not args.loop_stages and not args.stages and not args.channels,PREFETCH_PAGES=distance if cache else 0,
                                         num_warps=4,num_stages=args.stages if cache and args.stages else 2,**(dict(VD=args.channels) if args.channels and cache else {}),**(dict(maxnreg=args.max_reg) if cache and args.max_reg else {}))
            compiled=call()
            resources.append(dict(registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared))
            graph,_=capture(call)
            graphs.append(graph);outputs.append((out,lse))
        times=[[],[]]
        for rep in range(32):
            for i in (rep%2,1-rep%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(10):graphs[i].replay()
                b.record();b.synchronize()
                if rep>=2:times[i].append(a.elapsed_time(b)/10)
        ms=list(map(statistics.median,times))
        checks=[torch.equal(a,b) for a,b in zip(*outputs)]
        results.append(dict(pages=pages,ms=ms,speedup=ms[0]/ms[1],equal=checks,resources=resources,samples_ms=times))
        print(pages,ms,checks,resources,flush=True)
    path=root/(f'exact_channels_{args.channels}_v1' if args.channels else f'exact_stage_{args.stages}_reg_{args.max_reg}_v1' if args.stages else f'exact_unroll_{args.unroll}_loop_{args.loop_stages}_v1' if args.unroll or args.loop_stages else f'exact_query_group_{group}_v1' if group else f'exact_prefetch_{distance}_v1' if distance else 'exact_cached_lut_v1');path.mkdir(exist_ok=True)
    (path/'module.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
