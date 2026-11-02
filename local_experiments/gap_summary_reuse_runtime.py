"""Share identical FP32 K/V conversions between current pooling and covariance."""
import torch,time
from .gap_group_runtime import GapGroupRuntime
class GapSummaryReuseRuntime(GapGroupRuntime):
    @torch.no_grad()
    def current_summary(self,l):
        p=self.n[l];key=(l,p);cache=self.engine.cache[l];m=self.model[l]
        if key not in self.pool_graphs:
            k=cache['k'][0,p*64:p*64+4608].transpose(0,1).reshape(12,72,64,128)
            v=cache['v'][0,p*64:p*64+4608].transpose(0,1).reshape_as(k)
            def run():
                kf=k.float();vf=v.float()
                pk,base,detail=m.pool(kf,vf,cheap=True);cov=m.covariance(kf,vf)
                for dest,src in zip(self.current_outputs,(pk,base,detail,cov)):dest.copy_(src)
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):run()
            torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.summary_pool):run()
            self.pool_graphs[key]=graph;self.pool_capture_seconds+=time.perf_counter()-start
        self.pool_graphs[key].replay();return self.current_outputs
    def finish(self):return super().finish()|dict(current_conversion='one shared FP32 conversion per K/V, preserves original strides')
