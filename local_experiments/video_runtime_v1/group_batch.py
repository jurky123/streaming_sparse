"""Batch independent clean-layer partition trees within one clean forward."""
import torch
from .grouping_dataflow import balanced,group_features
from .group_gather import gather_pair
from ..cache_grouping import reorder

class BatchedGrouping:
    def __init__(self,config,one_hot=True):
        self.config=config
        self.one_hot=one_hot
        self.entries={}

    def compute(self,inputs,page_order):
        features=[]
        for model,k,v in inputs:
            kh,vh=(x[0].transpose(0,1) for x in (k,v))
            f=group_features(model,kh.float()[:,None],vh.float()[:,None],True)
            features.append(f.squeeze(1))
        x=torch.cat(features)
        h,n,d=x.shape
        order=balanced(x,[3,3,2,2,2],self.one_hot)
        xp=reorder(x,order)
        inner=balanced(xp.reshape(-1,64,d),[2,2],self.one_hot).reshape(h,n//64,64)
        order=order.reshape_as(inner).gather(2,inner)[:,page_order].flatten(1)
        outputs=[]
        for (_,k,v),idx in zip(inputs,order.split(12)):
            kg,vg=gather_pair(k[0].transpose(0,1),v[0].transpose(0,1),idx)
            outputs.append((kg.reshape(12,72,64,128),vg.reshape(12,72,64,128),idx))
        return outputs

    def __call__(self,inputs):
        # Weights are immutable in the inference runtime. Keep the exact models
        # used by each captured graph alive along with its input buffers.
        key=tuple((id(m),k.stride(),v.stride()) for m,k,v in inputs)
        if key not in self.entries:
            from .profile_route import capture
            buffers=[(m,k.clone(),v.clone()) for m,k,v in inputs]
            pages=self.config.page_order(inputs[0][1].device)
            graph,outputs=capture(lambda:self.compute(buffers,pages))
            self.entries[key]=(graph,buffers,outputs,pages)
        graph,buffers,outputs,_=self.entries[key]
        for (_,bk,bv),(_,k,v) in zip(buffers,inputs):
            bk.copy_(k)
            bv.copy_(v)
        graph.replay()
        return outputs

class ParallelGrouping:
    """Run independent copies of the accepted graph; join before consuming outputs."""
    def __init__(self,config,width,exact_small=False,fused_norm=False,fused_features=False,staged_softmax=False):
        from .grouping_dataflow import GraphGrouping
        self.streams=[torch.cuda.Stream() for _ in range(width)]
        self.groupers=[GraphGrouping(config,spatial_softmax=True,paired_gather=True,one_hot=True,exact_small=exact_small,fused_norm=fused_norm,fused_features=fused_features,staged_softmax=staged_softmax) for _ in self.streams]

    def __call__(self,inputs):
        main=torch.cuda.current_stream()
        outputs=[]
        for (model,k,v),stream,group in zip(inputs,self.streams,self.groupers):
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                outputs.append(group(model,k,v))
                k.record_stream(stream)
                v.record_stream(stream)
        for stream in self.streams[:len(inputs)]:main.wait_stream(stream)
        return outputs
