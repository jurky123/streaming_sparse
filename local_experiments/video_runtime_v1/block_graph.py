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
        correct,heads=self.runtime.correction_scope(l,cur['step'])
        # Head tensors are immutable between scope changes; retain them with the graph.
        cross=kwargs['crossattn_cache']
        if 'cross' not in buf:
            buf['cross']=dict(k=torch.empty_like(cross['k']),v=torch.empty_like(cross['v']),is_init=True)
            buf['cross_source']=None
        if buf['cross_source'] is None or any(buf['cross_source'][i] is not cross[n] for i,n in enumerate(('k','v'))):
            for name in ('k','v'):buf['cross'][name].copy_(cross[name])
            buf['cross_source']=(cross['k'],cross['v'])
        key=(l,chunk,self.runtime.sparse,correct,None if heads is None else heads.data_ptr())
        if key not in self.entries:
            static=dict(kwargs,e=buf['e'],context=buf['context'],crossattn_cache=buf['cross'])
            def call():buf['out'].copy_(self.original[l](buf['x'],**static))
            # Initialize nested route/tail graphs before the outer capture.
            call()
            previous_inline=self.runtime.batch_graph.inline
            self.runtime.batch_graph.inline=True
            try:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):call()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=self.pool):call()
            finally:
                self.runtime.batch_graph.inline=previous_inline
            self.entries[key]=(graph,heads)
        self.engine.current['layer']=l
        self.entries[key][0].replay()
        return buf['out']
    def close(self):
        for l,fn in self.original.items():self.engine.wrapper.model.blocks[l].forward=fn
