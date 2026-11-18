import argparse
import json
import statistics
import torch
from . import grouping_dataflow as grouping
from .weights import ROOT
from ..review_v1.model import Layer
from ..review_v1.ops import Config

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--paired-gather',action='store_true')
    parser.add_argument('--one-hot-gather',action='store_true')
    args=parser.parse_args()
    gather=args.paired_gather
    one_hot=args.one_hot_gather
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'grouping_dataflow_v1/input.pt',weights_only=True)
    model=Layer({n:v.cuda() for n,v in data['model'].items()}).eval()
    k,v=data['k'].cuda(),data['v'].cuda()
    graphs=[];outputs=[];samples=[[],[]]
    for enabled in (False,True):
        grouping.FUSED_SMALL_DIRECTION=enabled and not (gather or one_hot)
        grouping.PAIRED_GATHER=enabled and gather
        grouping.ONE_HOT_GATHER=enabled and one_hot
        fn=grouping.GraphGrouping(Config(history_pages=360),spatial_softmax=True)
        outputs.append(fn(model,k,v));graphs.append(fn)
    equal=[torch.equal(a,b) for a,b in zip(*outputs)]
    if gather or one_hot: assert all(equal)
    a,b=(o[2] for o in outputs)
    order_match=(a==b).float().mean().item()
    perms=all(torch.equal(order.sort(-1).values,torch.arange(4608,device=k.device).expand(12,-1)) for order in (a,b))
    pages=(a.reshape(12,72,64).sort(-1).values==b.reshape(12,72,64).sort(-1).values).all(-1).float().mean().item()
    for r in range(20):
        for i in (r%2,1-r%2):
            start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(20):graphs[i](model,k,v)
            end.record();end.synchronize()
            samples[i].append(start.elapsed_time(end)/20)
    ms=list(map(statistics.median,samples))
    result=dict(equal=equal,ms=ms,speedup=ms[0]/ms[1],order_match=order_match,permutations=perms,page_sets_equal_fraction=pages,samples_ms=samples)
    out=root/('one_hot_gather_v1' if one_hot else 'group_gather_v1' if gather else 'group_direction_v1');out.mkdir(exist_ok=True)
    (out/'module.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!='samples_ms'})
if __name__=='__main__':main()
