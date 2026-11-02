"""Experimental whole-current-summary compilation; no backbone or history changes."""
import time,torch
from .gap_ordered_runtime import GapFlatOrderedRuntime
from .gap_cached_ops import cached_pool
def summary(m,k,v):
    k=k.float();v=v.float();pk,base,detail=cached_pool(m,k,v,cheap=True)
    return pk,base,detail,m.covariance(k,v)
compiled_summary=torch.compile(summary,fullgraph=True,options={'triton.cudagraphs':False})
class GapPoolCompiledRuntime(GapFlatOrderedRuntime):
    @torch.no_grad()
    def current_summary(self,l):
        p=self.n[l];key=(l,p);cache=self.engine.cache[l];m=self.model[l]
        if key not in self.pool_graphs:
            k=cache['k'][0,p*64:p*64+4608].transpose(0,1).reshape(12,72,64,128)
            v=cache['v'][0,p*64:p*64+4608].transpose(0,1).reshape_as(k)
            def run():
                pk,base,detail,cov=compiled_summary(m,k,v)
                for dest,src in zip(self.current_outputs,(pk,base,detail,cov)):dest.copy_(src)
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):run()
            torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.summary_pool):run()
            self.pool_graphs[key]=graph;self.pool_capture_seconds+=time.perf_counter()-start
        self.pool_graphs[key].replay();return self.current_outputs
    def finish(self):return super().finish()|dict(current_pool='compiled native summary, unchanged algorithms; floating arithmetic may differ')
