"""Full route graph with explicit per-call input staging; no retained KV summaries."""
import math,time,torch
from .gap_pool_linear_runtime import GapPoolLinearRuntime
from .gap_route_input_kernels import query, kvar
from .gap_route_fused_tuned_runtime import project_stats
from .gap_route_fused_stats_ops import postprocess
from .gap_route_staging import stage
class GapRouteStagedGraphRuntime(GapPoolLinearRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.route_complete_graphs={};self.route_complete_outputs={}
        self.route_q=torch.empty(1,4608,12,128,device='cuda',dtype=torch.bfloat16)
        self.route_pk=torch.empty(12*(self.cap+72)*4*128,device='cuda')
        self.route_basis=torch.empty(12,128,8,device='cuda')
    @torch.no_grad()
    def decision(self,l,q,pk,base,detail,cov,vvar):
        if self.training_event:return super().decision(l,q,pk,base,detail,cov,vvar)
        m=self.model[l];h,p,_,d=pk.shape;t=q.shape[1]//64;n=self.probe_count
        kb=self.route_pk[:pk.numel()].view_as(pk);bb=self.gbase[:base.numel()].view_as(base);db=self.gdetail[:detail.numel()].view_as(detail);cb=self.gcov[:cov.numel()].view_as(cov);vb=self.gvar[:vvar.numel()].view_as(vvar)
        assert q.shape==self.route_q.shape and q.dtype==self.route_q.dtype
        stage([(self.route_q,q),(kb,pk),(bb,base),(db,detail),(cb,cov),(vb,vvar),(self.gw,m.route),(self.ggate,m.residual_gate),(self.route_basis,m.basis)])
        if p not in self.route_complete_graphs:
            raw=self.graw[:h*t*n*p*4].view(h,t*n,p*4);z=self.route_logz[:h*t*n*p].view(h,t,n,p)
            def run():
                selected=query(self.route_q,self.probe_offsets)
                project_stats(selected.reshape(h,t*n,d),kb.flatten(1,2),z,raw)
                qp=selected@self.route_basis.to(torch.bfloat16).float()[:,None]/math.sqrt(d)
                return postprocess(raw.view(h,t,n,p,4),qp,bb,db,cb,vb,kvar(kb),self.gw,self.ggate,self.ghaar,z)
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=run()
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.graph_pool):out=run()
            self.route_complete_graphs[p]=graph;self.route_complete_outputs[p]=out;self.capture_seconds+=time.perf_counter()-start
        self.route_complete_graphs[p].replay();return self.route_complete_outputs[p]
    def finish(self):return super().finish()|dict(route_full_graph=True,route_staging_workspace_bytes=sum(x.numel()*x.element_size() for x in [self.route_q,self.route_pk,self.route_basis]),route_staging_note='overwritten every decision; inference workspace, no reused historical summaries; copies included')
