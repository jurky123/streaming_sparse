"""Reuse one CUDA graph per history length and router kind, across all layers."""
import copy
from types import SimpleNamespace
import torch
from .core import Sketch


class RouteGraph:
    def __init__(self,route,q,sketch,prepared,head,beta,kind,batch,pool):
        self.q=torch.empty_like(q,memory_format=torch.contiguous_format)
        self.prepared=(prepared[0].clone(),[(k.clone(),c.clone(),v.clone()if v is not None else None)for k,c,v in prepared[1]])
        self.sketch=Sketch([],sketch.costs.clone(),sketch.origins.clone(),sketch.history_tokens,
            sketch.current_tokens,sketch.key_basis,sketch.value_basis)
        self.head=copy.deepcopy(head)if head is not None else None
        self.router=SimpleNamespace(kind=kind,beta=beta,heads={f'0_{kind}':self.head})
        self.step=q.new_zeros((),dtype=torch.float32)
        self.q.copy_(q)
        def run():
            masks=[route(0,self.q[:,offset:offset+batch],self.sketch,self.prepared,self.step,self.router)
                   for offset in range(0,self.q.shape[1],batch)]
            return torch.cat(masks,1)
        # Inductor compilation / cuBLAS handle initialization must precede capture.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):run()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,pool=pool):self.output=run()

    def __call__(self,q,prepared,head,step):
        self.q.copy_(q);self.prepared[0].copy_(prepared[0]);self.step.fill_(step)
        for stored,new in zip(self.prepared[1],prepared[1]):
            for a,b in zip(stored,new):
                if a is not None:a.copy_(b)
        if self.head is not None:
            for a,b in zip(self.head.parameters(),head.parameters()):a.copy_(b)
            for a,b in zip(self.head.buffers(),head.buffers()):a.copy_(b)
        self.graph.replay()
        return self.output


class GraphCache:
    def __init__(self,route,batch):
        self.route=route;self.batch=batch;self.graphs={};self.pool=torch.cuda.graph_pool_handle()

    def __call__(self,layer,q,sketch,prepared,step,router):
        key=(router.kind,sketch.history_tokens,q.shape[1])
        head=None if router.kind=='U-Sol'else router.heads[f'{layer}_{router.kind}']
        if key not in self.graphs:
            print(f'CUDA graph warmup/capture: {key}',flush=True)
            self.graphs[key]=RouteGraph(self.route,q,sketch,prepared,head,router.beta,router.kind,self.batch,self.pool)
        return self.graphs[key](q,prepared,head,step)
