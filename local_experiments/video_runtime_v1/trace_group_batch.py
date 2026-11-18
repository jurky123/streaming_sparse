"""Locate the first batch-size-dependent operation in the original partition tree."""
import json
import torch
from .weights import ROOT
from .grouping_dataflow import group_features
from ..review_v1.model import Layer

def trace(x,factors,width):
    h,n,d=x.shape
    groups=1
    for factor in factors:
        size=n//groups
        x=x.reshape(h*groups,size,d)
        xc=x-x.mean(1,keepdim=True)
        var=xc.square().mean(1)
        axis=var.argmax(-1,keepdim=True)
        proj=xc.gather(2,axis[:,None].expand(-1,size,-1))
        for label,value in [('center',xc),('variance',var),('axis',axis)]:
            yield f'{size}/{label}',value
        for it in range(4):
            direction=torch.bmm(xc.transpose(1,2),proj).squeeze(-1)
            yield f'{size}/transpose-{it}',direction
            direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            yield f'{size}/direction-{it}',direction
            proj=torch.bmm(xc,direction.unsqueeze(-1))
            yield f'{size}/projection-{it}',proj
        order=proj.squeeze(-1).argsort(dim=-1,stable=True)
        yield f'{size}/sort',order
        x=x.gather(1,order[:,:,None].expand_as(x))
        groups*=factor

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'grouping_dataflow_v1/input.pt',weights_only=True)
    model=Layer({n:v.cuda() for n,v in data['model'].items()}).eval()
    k,v=(data[n].cuda()[0].transpose(0,1)[:,None].float() for n in ('k','v'))
    f=group_features(model,k,v,True).squeeze(1)
    xs=[f.roll(i*37,1) for i in range(4)]
    results={}
    for inner in (False,True):
        inputs=[x.reshape(-1,64,32) if inner else x for x in xs]
        factors=[2,2] if inner else [3,3,2,2,2]
        separate=[trace(x,factors,1) for x in inputs]
        batched=trace(torch.cat(inputs),factors,4)
        for (name,new),*refs in zip(batched,*separate):
            old=torch.cat([v for _,v in refs])
            mismatch=(old!=new).sum().item()
            results[name]=dict(different=mismatch,total=old.numel(),max_abs=(old-new).abs().max().item())
    path=root/'batched_balanced_v1/op_compare.json'
    path.write_text(json.dumps(results,indent=2))
    print([(k,v['different']) for k,v in results.items() if v['different']][:16])
if __name__=='__main__':main()
