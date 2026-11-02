"""Native-order feature-axis parallel softmax in cached history grouping."""
import math,types,torch
from .gap_gain_runtime import GapGainRuntime
from .gap_spatial_softmax import spatial
def group_features(self,k,v):
    # H,P,T,D; P=1 for physical chunk grouping, P=pages for local pooling.
    kc=k-k.mean(-2,keepdim=True);score=torch.einsum('hptd,hrd->hptr',kc,self.bank)/math.sqrt(128)
    rms=score.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6);key=score/rms
    a=spatial(score,block=4 if k.shape[-2]>64 else 8);vp=torch.einsum('hptd,hdr->hptr',v,self.group_v);mean=(a*vp).sum(-2,keepdim=True)
    response=a*k.shape[-2]*(vp-mean);response=response/response.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    f=key+.1*self.group_gate.tanh()[:,None,None]*response
    return f

class GapSpatialRuntime(GapGainRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for m in self.model.layers.values():m.group_features=types.MethodType(group_features,m)
    def finish(self):return super().finish()|dict(group_softmax='native-order spatial softmax with feature-axis parallelism')
