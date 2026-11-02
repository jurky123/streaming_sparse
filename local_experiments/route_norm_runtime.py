import math
import torch
from .route_norm_kernels import logz
from .route_fused_kernels import score_lut
from .route_packed_runtime import PackedRouter,PackedRuntime
from .high_sparse_gain_runtime import GainRuntime
def norm_postprocess(raw,code,vvar,kvar,w,p,keep,enabled):
    prob=logz(raw,p).softmax(-1);mass=prob.mean(2)
    own=(torch.arange(72,device=raw.device)+p-72)[None].expand(12,-1)
    z=mass.clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    if not enabled:return ids.sort(-1).values[None].int(),None
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=weighted@code.float()/den;allvalue=mass@code.float()
    return score_lut(prob,mass,code,vvar,kvar,w,mask,den,selected,allvalue,ids),None


class NormRouter(PackedRouter):
    postprocess=staticmethod(norm_postprocess)
class NormRuntime(PackedRuntime):
    def __init__(self,engine,checkpoint,strength=1.):
        GainRuntime.__init__(self,engine,checkpoint,strength)
        if not hasattr(engine,'shared_norm_router'):engine.shared_norm_router=NormRouter()
        self.router=engine.shared_norm_router;self.persistent_bytes+=self.router.kv_aux_bytes
        assert self.persistent_bytes<=self.budget;self.route_meta=None
    def finish(self):return super().finish()|dict(route_implementation='fused native-order 4-representative logsumexp; native page softmax/topk; fused features-score/swap; captured QK/variance; shared calculated packed-K')
