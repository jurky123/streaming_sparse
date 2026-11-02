"""Moment-preserving pooling, cheap current pages, cached within-group KV covariance."""
import torch
from torch import nn
from .all_sparse_model import SparseLayer
from .conditional_v_kernels import HAAR
class GapLayer(SparseLayer):
    def __init__(self,rt,l):
        super().__init__(rt,l)
        del self.comp_logscale
        self.register_buffer('comp_logscale',torch.zeros(12,device='cuda'))
        self.residual_gate=nn.Parameter(torch.full((12,),.1,device='cuda'))
    def pool(self,k,v,cheap=False):
        h,p,_,d=k.shape;k=k.float();v=v.float()
        if cheap:
            kc=k-k.mean(-2,keepdim=True)
            f=torch.einsum('hptd,hrd->hptr',kc,self.bank)/128**.5
            f=f/f.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        else:f=self.group_features(k,v)
        f=f.reshape(h,p,4,16,32);kg=k.reshape(h,p,4,16,d);vg=v.reshape_as(kg)
        z=torch.einsum('hprtd,hrd->hprt',kg-kg.mean(-2,keepdim=True),self.pool_k)
        z=z+torch.einsum('hprtd,hrd->hprt',vg-vg.mean(-2,keepdim=True),self.pool_v)+torch.einsum('hprtf,hrf->hprt',f,self.pool_feature)
        z=z-z.mean(-1,keepdim=True);a=(2*torch.tanh(z/2)).softmax(-1)
        pk=(a[...,None]*kg).sum(-2);pv=(a[...,None]*vg).sum(-2)
        # Preserve page-level K mean and encoded V mean. No new cache bytes.
        pk=pk-pk.mean(2,keepdim=True)+k.mean(2,keepdim=True)
        basis=self.basis.to(torch.bfloat16).float()
        codes=torch.einsum('hprd,hdk->hprk',pv-self.center[:,None,None],basis)
        base=torch.einsum('hpd,hdk->hpk',v.mean(2)-self.center[:,None],basis).to(torch.bfloat16)
        haar=torch.tensor(HAAR,device=k.device,dtype=torch.float32)
        detail=(torch.einsum('kr,hprd->hpkd',haar,codes)/4).to(torch.bfloat16)
        return pk.contiguous(),base.contiguous(),detail.contiguous()
    def covariance(self,k,v):
        # One rank8 diagonal within-subgroup cross moment per physical page.
        basis=self.basis.to(torch.bfloat16).float();h,p,t,d=k.shape
        a=torch.einsum('hptd,hdr->hptr',k.float(),basis).reshape(h,p,4,16,8)
        b=torch.einsum('hptd,hdr->hptr',v.float(),basis).reshape_as(a)
        return ((a-a.mean(3,keepdim=True))*(b-b.mean(3,keepdim=True))).mean((2,3)).to(torch.bfloat16).contiguous()
class GapModel(nn.Module):
    def __init__(self,rt):super().__init__();self.layers=nn.ModuleDict({str(l):GapLayer(rt,l) for l in range(1,30)})
    def __getitem__(self,l):return self.layers[str(l)]
    @torch.no_grad()
    def project(self):
        for m in self.layers.values():
            m.pool_k.clamp_(-3,3);m.pool_v.clamp_(-.5,.5);m.pool_feature.clamp_(-.25,.25);m.route.clamp_(-3,3);m.group_gate.clamp_(-2,2);m.residual_gate.clamp_(-2,2)
