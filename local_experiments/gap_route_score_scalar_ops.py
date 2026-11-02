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
from .gap_route_score_scalar import _score_tiled
from .gap_conditional_kernel import conditional as decode_conditional
def postprocess(logits,qp,base,detail,cov,vvar,kvar,w,gate,haar,training=False):
    h,t,n,p,_=logits.shape;keep=int(math.floor(p*.1+1e-6))
    logz,within=stats(logits,order=1,div=1,exp=0);prob=logz.softmax(-1);mass,peak,spread=probability_stats(prob)
    weight=ordered_weight(prob,within,mass,block=64)
    conditional=decode_conditional(weight,base,detail,native_layout=True)
    # First-order diagonal projected KV-covariance proxy, not an exact page oracle.
    qnumerator=torch.einsum('htmp,htmd->htpd',prob,qp)
    own=(torch.arange(t,device=logits.device)+p-t)[None].expand(h,-1)
    z=mass.detach().clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=(weighted[...,None]*conditional).sum(2)/den;allvalue=(mass[...,None]*conditional).sum(2)
    s=torch.empty(h,t,p,device=logits.device)
    _score_tiled[(h*t,tr.cdiv(p,128))](peak,mass,conditional,vvar,kvar,w,mask,den,selected,allvalue,(mass/den).clamp(0,2),spread,s,p,128,*conditional.stride()[:3],num_warps=4,enable_fp_fusion=False)
    contribution,error=fused_contribution(qnumerator,mass,cov,mask)
    s=gain_finish(contribution,error,mask,s,gate)
    assert not training
    chosen=torch.empty(1,h,t,keep,device=logits.device,dtype=torch.int32)
    _lut[(h*t,)](s,mask,ids,chosen,p,keep,tr.next_power_of_2(p),tr.next_power_of_2(keep),num_warps=4)
    return chosen,None,None,None
