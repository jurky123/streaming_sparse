"""Measure independent clean grouping and next-block FFN work on two streams."""
import json
import statistics
import torch
from .weights import ROOT
from .grouping_dataflow import GraphGrouping
from ..review_v1.model import Layer
from ..review_v1.ops import Config

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'grouping_dataflow_v1/input.pt',weights_only=True)
    model=Layer({n:v.cuda() for n,v in data['model'].items()}).eval()
    k,v=data['k'].cuda(),data['v'].cuda()
    grouping=GraphGrouping(Config(history_pages=360),spatial_softmax=True)
    grouping(model,k,v)
    x=torch.randn(4608,1536,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(1536,8960,device='cuda',dtype=x.dtype)
    y=torch.empty(4608,8960,device='cuda',dtype=x.dtype)
    z=torch.empty_like(x)
    side=torch.cuda.Stream()
    main_stream=torch.cuda.current_stream()
    def run(overlap):
        stream=side if overlap else main_stream
        stream.wait_stream(main_stream)
        with torch.cuda.stream(stream):
            result=grouping(model,k,v)
        torch.mm(x,w,out=y)
        torch.mm(y,w.T,out=z)
        main_stream.wait_stream(stream)
        return result
    times=[[],[]]
    for rep in range(32):
        for i in (rep%2,1-rep%2):
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            result=run(i)
            b.record();b.synchronize()
            if rep==0 and i==0:
                reference=[t.clone() for t in (*result,z)]
            if rep==1:
                assert all(torch.equal(t,u) for t,u in zip(reference,(*result,z)))
            if rep>=2:times[i].append(a.elapsed_time(b))
    med=list(map(statistics.median,times))
    out=root/'clean_overlap_v1';out.mkdir(exist_ok=True)
    (out/'module.json').write_text(json.dumps(dict(medians=med,speedup=med[0]/med[1],samples_ms=times),indent=2))
    print(med,med[0]/med[1])
if __name__=='__main__':main()
