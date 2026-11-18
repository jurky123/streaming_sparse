import argparse,json,statistics
import torch
from .weights import ROOT
from .group_batch import BatchedGrouping
from .grouping_dataflow import GraphGrouping
from ..review_v1.model import Layer
from ..review_v1.ops import Config

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--streams',action='store_true')
    streams=parser.parse_args().streams
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'grouping_dataflow_v1/input.pt',weights_only=True)
    m=Layer({n:v.cuda() for n,v in data['model'].items()}).eval()
    k,v=(data[n].cuda() for n in ('k','v'))
    rows=[]
    for layers in (2,4,8):
        inputs=[(m,k.roll(37*i,1),v.roll(37*i,1)) for i in range(layers)]
        cfg=Config(history_pages=360)
        single=GraphGrouping(cfg,spatial_softmax=True,paired_gather=True,one_hot=True)
        if streams:
            from .group_batch import ParallelGrouping
            batch=ParallelGrouping(cfg,layers)
        else:batch=BatchedGrouping(cfg)
        outputs=batch(inputs)
        equal=[all(torch.equal(a,b) for a,b in zip(single(*inp),out)) for inp,out in zip(inputs,outputs)]
        samples=[[],[]]
        for rep in range(24):
            for i in (rep%2,1-rep%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(4):
                    if i:batch(inputs)
                    else:
                        for inp in inputs:single(*inp)
                b.record();b.synchronize()
                if rep>=4:samples[i].append(a.elapsed_time(b)/4/layers)
        ms=list(map(statistics.median,samples))
        rows.append(dict(layers=layers,equal=equal,ms_per_layer=ms,speedup=ms[0]/ms[1],samples_ms=samples))
        print(layers,equal,ms,flush=True)
    (root/'batched_balanced_v1'/('stream_group.json' if streams else 'full_group.json')).write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
