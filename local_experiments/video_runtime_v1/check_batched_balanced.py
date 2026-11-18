"""Measure amortizing the two hierarchical partition trees over clean layers."""
import json
import statistics
import torch
from .weights import ROOT
from .profile_route import capture
from .grouping_dataflow import balanced, group_features
from ..review_v1.model import Layer

def orders(f):
    h,n,d=f.shape
    parent=balanced(f,[3,3,2,2,2])
    x=f.gather(1,parent[:,:,None].expand_as(f))
    inner=balanced(x.reshape(-1,64,d),[2,2]).reshape(h,n//64,64)
    return parent.reshape_as(inner).gather(2,inner)

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'grouping_dataflow_v1/input.pt',weights_only=True)
    model=Layer({k:v.cuda() for k,v in data['model'].items()}).eval()
    k,v=(data[n].cuda()[0].transpose(0,1)[:,None].float() for n in ('k','v'))
    f=group_features(model,k,v,True).squeeze(1)
    rows=[]
    for layers in (2,4,8):
        inputs=[f.roll(i*37,1) for i in range(layers)]
        variants=[capture(lambda:[orders(x) for x in inputs]),capture(lambda:orders(torch.cat(inputs)).split(f.shape[0]))]
        equal=[torch.equal(a,b) for a,b in zip(variants[0][1],variants[1][1])]
        times=[[],[]]
        for rep in range(24):
            for i in (rep%2,1-rep%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(5):variants[i][0].replay()
                b.record();b.synchronize()
                if rep>=4:times[i].append(a.elapsed_time(b)/5/layers)
        med=list(map(statistics.median,times))
        rows.append(dict(layers=layers,equal=equal,ms_per_layer=med,speedup=med[0]/med[1],samples_ms=times))
        print(layers,equal,med,flush=True)
    out=root/'batched_balanced_v1';out.mkdir(exist_ok=True)
    (out/'module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
