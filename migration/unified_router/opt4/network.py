"""Evaluation adapter for independently routed, compact statistics kernels."""
import copy
import os
from types import SimpleNamespace
import torch
from ..core import Sketch
from ..opt2.network import load_optimized
from .route import CompactRoute


class CompactGraph:
    def __init__(self,route,q,sketch,prepared,head,beta,kind,batch,pool):
        self.q=torch.empty_like(q,memory_format=torch.contiguous_format)
        self.prepared=(prepared[0].clone(),[(k.clone(),c.clone(),v.clone()if v is not None else None) for k,c,v in prepared[1]])
        self.sketch=Sketch([],sketch.costs.clone(),sketch.origins.clone(),sketch.history_tokens,
            sketch.current_tokens,sketch.key_basis,sketch.value_basis)
        self.head=copy.deepcopy(head)if head is not None else None
        self.router=SimpleNamespace(kind=kind,beta=beta,heads={f'0_{kind}':self.head})
        self.step=q.new_zeros((),dtype=torch.float32);self.q.copy_(q)
        def run():
            # Projection-derived auxiliary data is shared across query tiles,
            # recomputed on replay for the current layer, never stale-cached.
            state=route.begin(self.prepared,self.sketch)
            masks=[route.run(0,self.q[:,off:off+batch],self.sketch,state,self.step,self.router)
                   for off in range(0,self.q.shape[1],batch)]
            return torch.cat(masks,1)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):run()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,pool=pool):self.output=run()

    def __call__(self,q,prepared,head,step):
        self.q.copy_(q);self.prepared[0].copy_(prepared[0]);self.step.fill_(step)
        for old,new in zip(self.prepared[1],prepared[1]):
            for a,b in zip(old,new):
                if a is not None:a.copy_(b)
        if self.head is not None:
            for a,b in zip(self.head.parameters(),head.parameters()):a.copy_(b)
            for a,b in zip(self.head.buffers(),head.buffers()):a.copy_(b)
        self.graph.replay();return self.output


class CompactCache:
    def __init__(self,route,batch):
        self.route=route;self.batch=batch;self.graphs={};self.pool=torch.cuda.graph_pool_handle()

    def __call__(self,layer,q,sketch,prepared,step,router):
        key=(router.kind,sketch.history_tokens,q.shape[1])
        head=None if router.kind=='U-Sol'else router.heads[f'{layer}_{router.kind}']
        if key not in self.graphs:
            print(f'Compact CUDA graph capture: {key}',flush=True)
            self.graphs[key]=CompactGraph(self.route,q,sketch,prepared,head,router.beta,router.kind,self.batch,self.pool)
        return self.graphs[key](q,prepared,head,step)


def load_compact(directory,training_seed=0,method=None,precision=None,score_precision=None,gp=None,tile=None,**options):
    method=method or os.environ.get('SLIMAV_COMPACT_METHOD','streamgram')
    precision=precision or os.environ.get('SLIMAV_COMPACT_PRECISION','fp16')
    score_precision=score_precision or os.environ.get('SLIMAV_SCORE_PRECISION','tf32x3')
    gp=gp or int(os.environ.get('SLIMAV_PAGE_TILE','8'))
    tile=tile or int(os.environ.get('SLIMAV_QUERY_TILE','64'))
    net,m=load_optimized(directory,training_seed,**options)
    route=CompactRoute(method,precision,score_precision,gp,tile,net.route_batch)
    net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch)
    net.compact_options=dict(method=method,precision=precision,score_precision=score_precision,gp=gp,tile=tile)
    return net,m


def main():
    from .. import fast_bench
    fast_bench.load_fast=load_compact;fast_bench.main()


if __name__=='__main__':main()
