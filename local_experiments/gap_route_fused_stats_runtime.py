"""Experimental two-term TF32 route projection with configurable precision tail."""
import torch,time,math
from .gap_input_runtime import gather_queries,representative_kvar
from .gap_route_fused_stats_ops import postprocess
from .gap_tail_packed_runtime import GapTailPackedRuntime
from .gap_route_bf16_stats import project_stats
class GapRouteFusedStatsRuntime(GapTailPackedRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.route_logz=torch.empty(self.graw.numel()//4,device=self.graw.device,dtype=self.graw.dtype)
    @torch.no_grad()
    def decision(self,l,q,pk,base,detail,cov,vvar):
        if self.training_event:return super().decision(l,q,pk,base,detail,cov,vvar)
        m=self.model[l];h,p,_,d=pk.shape;t=q.shape[1]//64;n=self.probe_count
        query=gather_queries(q,self.probe_offsets)
        raw=self.graw[:h*t*n*p*4].view(h,t*n,p*4)
        z=self.route_logz[:h*t*n*p].view(h,t,n,p)
        project_stats(query.reshape(h,t*n,d),pk.flatten(1,2),z,raw)
        logits=raw.view(h,t,n,p,4)
        qp=query@m.basis.to(torch.bfloat16).float()[:,None]/math.sqrt(d);kvar=representative_kvar(pk)
        args=(self.graw[:logits.numel()].view_as(logits),self.gqp,self.gbase[:base.numel()].view_as(base),self.gdetail[:detail.numel()].view_as(detail),self.gcov[:cov.numel()].view_as(cov),self.gvar[:vvar.numel()].view_as(vvar),self.gkvar[:kvar.numel()].view_as(kvar),self.gw,self.ggate,self.ghaar,z)
        for dest,src in zip(args[1:-2],(qp,base,detail,cov,vvar,kvar,m.route,m.residual_gate)):dest.copy_(src)
        if p not in self.graphs_new:
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=postprocess(*args)
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.graph_pool):out=postprocess(*args)
            self.graphs_new[p]=graph;self.outputs_new[p]=out;self.capture_seconds+=time.perf_counter()-start
        self.graphs_new[p].replay();return self.outputs_new[p]
    def finish(self):
        return super().finish()|dict(route_projection="fused BF16 high/low projection and fast statistics; linear exact weights",route_logz_workspace_bytes=self.route_logz.numel()*self.route_logz.element_size())
