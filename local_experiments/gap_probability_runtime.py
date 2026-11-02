"""Inference candidate: fuse weights using the native four-accumulator mean order."""
import math,torch,triton as tr
from .gap_feature_kernel import _features_conditional
from .route_fused_kernels import _lut
from .high_sparse_gain import score
from .gap_vector_stats import stats
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
    _features_conditional[(h*t,)](peak,mass,conditional,vvar,kvar,w,mask,den,selected,allvalue,(mass/den).clamp(0,2),spread,feat,p,tr.next_power_of_2(p),*conditional.stride()[:3],num_warps=4,enable_fp_fusion=False)
    s=score(mass,feat,w)
    contribution,error=fused_contribution(qnumerator,mass,cov,mask)
    s=gain_finish(contribution,error,mask,s,gate)
    assert not training
    chosen=torch.empty(1,h,t,keep,device=logits.device,dtype=torch.int32)
    _lut[(h*t,)](s,mask,ids,chosen,p,keep,tr.next_power_of_2(p),tr.next_power_of_2(keep),num_warps=4)
    return chosen,None,None,None

class GapProbabilityRuntime(GapSpatialRuntime):
    def decision(self,*args,**kwargs):
        original=module.postprocess
        try:
            module.postprocess=postprocess
            return GapVectorRuntime.decision(self,*args,**kwargs)
        finally:module.postprocess=original
    def finish(self):return super().finish()|dict(residual_gain='fused elementwise contributions/gain; native page sum and GEMM')
