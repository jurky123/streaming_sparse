import torch
from local_experiments.video_runtime_v1.grouping_dataflow import balanced,group_features

def arrange(rt,cell,model,k,v,learned):
    if cell=='GG':return learned
    kh,vh=[t[0].transpose(0,1) for t in (k,v)]
    ids=torch.arange(4608,device=k.device).expand(12,-1).reshape(12,72,64)
    if cell=='GN':ids=learned[2].reshape_as(ids).sort(-1).values
    if cell=='NG':
        g=rt.batched_group_original.groupers[0]
        f=group_features(model,kh.float()[:,None],vh.float()[:,None],g.spatial_softmax,g.fused_features,g.staged_softmax)
        inner=balanced(f.reshape(-1,64,32),[2,2],True,g.exact_small,g.fused_norm,g.fused_sort)
        ids=ids.gather(2,inner.reshape_as(ids))
    order=ids.reshape(12,-1)
    def take(t):return t.gather(1,order[...,None].expand(-1,-1,128)).reshape(12,72,64,128)
    return take(kh),take(vh),order

def install(rt,cell):
    old=rt.batched_group
    rt.batched_group_original=old
    def group(inputs):
        if rt.engine.current['chunk']==0 or cell=='GG':return old(inputs)
        source=old(inputs) if cell=='GN' else [None]*len(inputs)
        return [arrange(rt,cell,*args,s) for args,s in zip(inputs,source)]
    rt.batched_group=group
