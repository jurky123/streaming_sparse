"""Replay complete noisy transformer blocks; clean commits remain eager.

Only layers already using the sparse runtime are wrapped. Each chunk has fixed
RoPE/history addresses; timestep modulation and x are copied into shared inputs.
"""
import torch

class BlockGraphs:
    def __init__(self,engine,runtime,layers=range(1,30)):
        self.runtime=runtime;self.engine=engine;self.original={};self.entries={};self.buffers={};self.pool=torch.cuda.graph_pool_handle()
        for l in layers:
            block=engine.wrapper.model.blocks[l];self.original[l]=block.forward
            def forward(x,index=l,**kwargs):return self.forward(index,x,kwargs)
            block.forward=forward
    @torch.no_grad()
    def forward(self,l,x,kwargs):
        cur=self.engine.current;chunk=cur['chunk']
        if cur['step']==4 or chunk<2:return self.original[l](x,**kwargs)
        if l not in self.buffers:
            self.buffers[l]=dict(x=torch.empty_like(x),out=torch.empty_like(x),
                                 e=torch.empty_like(kwargs['e']),context=torch.empty_like(kwargs['context']))
        buf=self.buffers[l];buf['x'].copy_(x)
        for name in ('e','context'):buf[name].copy_(kwargs[name])
        key=(l,chunk)
        if key not in self.entries:
            static=dict(kwargs,e=buf['e'],context=buf['context'])
            def call():buf['out'].copy_(self.original[l](buf['x'],**static))
            # Initialize nested route/tail graphs before the outer capture.
            call()
            self.runtime.batch_graph.inline=True;self.runtime.inline_summary=True
            try:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):call()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=self.pool):call()
            finally:
                self.runtime.batch_graph.inline=False;self.runtime.inline_summary=False
            self.entries[key]=graph
        self.engine.current['layer']=l
        self.entries[key].replay()
        return buf['out']
    def close(self):
        for l,fn in self.original.items():self.engine.wrapper.model.blocks[l].forward=fn
