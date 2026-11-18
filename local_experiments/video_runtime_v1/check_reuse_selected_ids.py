"""Measure the no-correction route with one ranking at the original capacity."""
import argparse
import json
import statistics
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights
from .pipeline_graph import BatchGraphs
from .risk_projected import risk_inline
from .profile_route import capture
from .weights import ROOT
from ..review_v1.ops import Summary,Config

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--compact",action="store_true")
    ap.add_argument("--ties",action="store_true")
    args=ap.parse_args()
    compact=args.compact
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    model,_,_,_,_=load_weights(cfg)
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'tail_liveness_v1/input.pt',weights_only=True)
    offsets=torch.linspace(0,63,16,device='cuda').round().long()
    q=data['q'].cuda()[0].transpose(0,1).reshape(12,72,64,128).index_select(2,offsets)
    rows=[]
    torch._dynamo.config.cache_size_limit=64
    for p in (216,288,360,432):
        s=Summary(*(data['summary'][name][:,:p].cuda().contiguous() for name in ('keys','mean','detail','stats')))
        e=dict(q=q,s=s,en={},theta=model[15].theta,basis=model[15].basis,
               own=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1))
        rt=BatchGraphs(risk_inline,None,Config(history_pages=360),compile_route=True,probes_only=True,input_m16=True,static_route=True)
        rt.scale_in_qk=True
        variants=[]
        for reuse in (False,"ties" if args.ties else "compact" if compact else True):
            rt.reuse_selected_ids=reuse
            variants.append(capture(lambda:rt.compute(e,False)))
        samples=[[],[]]
        for rep in range(24):
            for i in (rep%2,1-rep%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(12):variants[i][0].replay()
                b.record();b.synchronize()
                if rep>=4:samples[i].append(a.elapsed_time(b)/12)
        ms=list(map(statistics.median,samples))
        before,after=[item[1] for item in variants]
        row=dict(pages=p,ms=ms,speedup=ms[0]/ms[1],equal=[torch.equal(x,y) for x,y in zip(before,after)],
                 reordered_rows=(before[0]!=after[0]).any(-1).sum().item())
        print(row,flush=True);rows.append(dict(**row,samples_ms=samples))
    out=root/('selected_tie_order_v1' if args.ties else 'compact_selected_order_v1' if compact else 'reuse_selected_ids_v1')
    out.mkdir(exist_ok=True)
    (out/'module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
