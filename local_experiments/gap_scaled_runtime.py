"""Inference candidate: fuse weights using the native four-accumulator mean order."""
import math,time,torch,triton as tr
from .gap_feature_tiled import _features_tiled as _features_conditional
from .route_fused_kernels import _lut
from .high_sparse_gain import score
from .gap_scaled_stats import scaled_stats as stats
from .gap_weight_ordered import weight as ordered_weight
from .gap_vector_runtime import GapVectorRuntime
from .gap_spatial_runtime import GapSpatialRuntime
from .gap_probability_stats import probability_stats
from .gap_gain_kernel import contribution as fused_contribution,finish as gain_finish
from . import gap_feature_runtime as module
def postprocess(logits,qp,base,detail,cov,vvar,kvar,w,gate,haar,training=False):
    h,t,n,p,_=logits.shape;keep=int(math.floor(p*.1+1e-6))
    logz,within=stats(logits,order=1,div=1,exp=0);prob=logz.softmax(-1);mass,peak,spread=probability_stats(prob)
    weight=ordered_weight(prob,within,mass,block=64)
    coefficients=torch.einsum('htpr,ar->htpa',weight,haar)
    conditional=base.float()[:,None]+torch.einsum('htpa,hpad->htpd',coefficients,detail.float())
    # First-order diagonal projected KV-covariance proxy, not an exact page oracle.
    qnumerator=torch.einsum('htmp,htmd->htpd',prob,qp)
    own=(torch.arange(t,device=logits.device)+p-t)[None].expand(h,-1)
    z=mass.detach().clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=(weighted[...,None]*conditional).sum(2)/den;allvalue=(mass[...,None]*conditional).sum(2)
    feat=torch.empty(h,t,p,14,device=logits.device)
    _features_conditional[(h*t,tr.cdiv(p,128))](peak,mass,conditional,vvar,kvar,w,mask,den,selected,allvalue,(mass/den).clamp(0,2),spread,feat,p,128,*conditional.stride()[:3],num_warps=4,enable_fp_fusion=False)
    s=score(mass,feat,w)
    contribution,error=fused_contribution(qnumerator,mass,cov,mask)
    s=gain_finish(contribution,error,mask,s,gate)
    assert not training
    chosen=torch.empty(1,h,t,keep,device=logits.device,dtype=torch.int32)
    _lut[(h*t,)](s,mask,ids,chosen,p,keep,tr.next_power_of_2(p),tr.next_power_of_2(keep),num_warps=4)
    return chosen,None,None,None

from .gap_probability_runtime import GapProbabilityRuntime
class GapScaledRuntime(GapProbabilityRuntime):
    @torch.no_grad()
    def decision(self,l,q,pk,base,detail,cov,vvar):
        if self.training_event:return super().decision(l,q,pk,base,detail,cov,vvar)
        m=self.model[l];h,p,_,d=pk.shape;t=q.shape[1]//64;n=self.probe_count
        query=q[0].transpose(0,1).float().reshape(h,t,64,d).index_select(2,self.probe_offsets)
        raw=self.graw[:h*t*n*p*4].view(h,t*n,p*4)
        torch.bmm(query.reshape(h,t*n,d),pk.flatten(1,2).transpose(1,2),out=raw)
        logits=raw.view(h,t,n,p,4)
        qp=query@m.basis.to(torch.bfloat16).float()[:,None]/math.sqrt(d);kvar=(pk-pk.mean(2,keepdim=True)).square().mean((2,3)).log1p()
        args=(self.graw[:logits.numel()].view_as(logits),self.gqp,self.gbase[:base.numel()].view_as(base),self.gdetail[:detail.numel()].view_as(detail),self.gcov[:cov.numel()].view_as(cov),self.gvar[:vvar.numel()].view_as(vvar),self.gkvar[:kvar.numel()].view_as(kvar),self.gw,self.ggate,self.ghaar)
        for dest,src in zip(args[1:-1],(qp,base,detail,cov,vvar,kvar,m.route,m.residual_gate)):dest.copy_(src)
        if p not in self.graphs_new:
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=postprocess(*args)
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.graph_pool):out=postprocess(*args)
            self.graphs_new[p]=graph;self.outputs_new[p]=out;self.capture_seconds+=time.perf_counter()-start
        self.graphs_new[p].replay();return self.outputs_new[p]
    def finish(self):return super().finish()|dict(route_scale='fused into representative normalization',feature_tile=128)
