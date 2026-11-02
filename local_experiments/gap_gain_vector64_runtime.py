"""Full route graph with explicit per-call input staging; no retained KV summaries."""
import math,time,torch
from .gap_route_qp_runtime import GapRouteQpRuntime
from .gap_route_query_project import gather_project
from .gap_route_input_kernels import query, kvar
from .gap_route_fused_tuned_runtime import project_stats
from functools import partial
from .gap_gain_project_ops import postprocess as _postprocess
from .gap_gain_project_vector import contribution
postprocess=partial(_postprocess,contribution_fn=partial(contribution,block=64))
from .gap_route_staging import stage
class GapGainVector64Runtime(GapRouteQpRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gain_project_graphs={};self.gain_project_outputs={}
    @torch.no_grad()
    def decision(self,l,q,pk,base,detail,cov,vvar):
        if self.training_event:return super().decision(l,q,pk,base,detail,cov,vvar)
        m=self.model[l];h,p,_,d=pk.shape;t=q.shape[1]//64;n=self.probe_count
        kb=self.route_pk[:pk.numel()].view_as(pk);bb=self.gbase[:base.numel()].view_as(base);db=self.gdetail[:detail.numel()].view_as(detail);cb=self.gcov[:cov.numel()].view_as(cov);vb=self.gvar[:vvar.numel()].view_as(vvar)
        assert q.shape==self.route_q.shape and q.dtype==self.route_q.dtype
        stage([(self.route_q,q),(kb,pk),(bb,base),(db,detail),(cb,cov),(vb,vvar),(self.gw,m.route),(self.ggate,m.residual_gate),(self.route_basis,m.basis)])
        if p not in self.gain_project_graphs:
            raw=self.graw[:h*t*n*p*4].view(h,t*n,p*4);z=self.route_logz[:h*t*n*p].view(h,t,n,p)
            def run():
                selected,qp=gather_project(self.route_q,self.probe_offsets,self.route_basis)
                project_stats(selected.reshape(h,t*n,d),kb.flatten(1,2),z,raw)
                return postprocess(raw.view(h,t,n,p,4),qp,bb,db,cb,vb,kvar(kb),self.gw,self.ggate,self.ghaar,z)
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=run()
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.graph_pool):out=run()
            self.gain_project_graphs[p]=graph;self.gain_project_outputs[p]=out;self.capture_seconds+=time.perf_counter()-start
        self.gain_project_graphs[p].replay();return self.gain_project_outputs[p]
    def finish(self):return super().finish()|dict(residual_projection="fused vector FP32 probe weighted projection and covariance contribution; approximate order",route_full_graph=True,route_staging_workspace_bytes=sum(x.numel()*x.element_size() for x in [self.route_q,self.route_pk,self.route_basis]),route_staging_note='overwritten every decision; inference workspace, no reused historical summaries; copies included')
