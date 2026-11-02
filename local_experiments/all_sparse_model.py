"""All sparse-module parameters. Hard physical pages, differentiable page-local pooling."""
import math
import torch
from torch import nn
from .conditional_v_kernels import HAAR
class SparseLayer(nn.Module):
    def __init__(self,rt,l):
        super().__init__()
        self.bank=nn.Parameter(rt.bank[l].float().clone())
        self.group_v=nn.Parameter((rt.bank[l].float()/rt.bank[l].float().norm(dim=-1,keepdim=True).clamp_min(1e-6)).transpose(1,2).contiguous())
        self.group_gate=nn.Parameter(torch.full((12,32),.1,device='cuda'))
        self.pool_k=nn.Parameter(rt.weights[l].float().clone());self.pool_v=nn.Parameter(torch.zeros_like(self.pool_k))
        self.pool_feature=nn.Parameter(torch.randn(12,4,32,device='cuda')*.001)
        self.basis=nn.Parameter(rt.basis[l].float().clone());self.center=nn.Parameter(rt.centers[l].float().clone())
        self.route=nn.Parameter(rt.sw[l].float().clone());self.comp_logscale=nn.Parameter(torch.zeros(12,device='cuda'))
    def group_features(self,k,v):
        # H,P,T,D; P=1 for physical chunk grouping, P=pages for local pooling.
        kc=k-k.mean(-2,keepdim=True);score=torch.einsum('hptd,hrd->hptr',kc,self.bank)/math.sqrt(128)
        rms=score.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6);key=score/rms
        a=score.softmax(-2);vp=torch.einsum('hptd,hdr->hptr',v,self.group_v);mean=(a*vp).sum(-2,keepdim=True)
        response=a*k.shape[-2]*(vp-mean);response=response/response.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        f=key+.1*self.group_gate.tanh()[:,None,None]*response
        return f
    def pool(self,k,v):
        h,p,_,d=k.shape;f=self.group_features(k.float(),v.float()).reshape(h,p,4,16,32)
        kg=k.float().reshape(h,p,4,16,d);vg=v.float().reshape_as(kg)
        score=torch.einsum('hprtd,hrd->hprt',kg-kg.mean(-2,keepdim=True),self.pool_k)
        score=score+torch.einsum('hprtd,hrd->hprt',vg-vg.mean(-2,keepdim=True),self.pool_v)+torch.einsum('hprtf,hrf->hprt',f,self.pool_feature)
        score=score-score.mean(-1,keepdim=True);a=(2*torch.tanh(score/2)).softmax(-1)
        pk=(a[...,None]*kg).sum(-2);pv=(a[...,None]*vg).sum(-2)
        basis=self.basis.to(torch.bfloat16).float();c=torch.einsum('hprd,hdk->hprk',pv-self.center[:,None,None],basis)
        base=c.mean(2).to(torch.bfloat16);haar=torch.tensor(HAAR,device=k.device,dtype=torch.float32)
        detail=(torch.einsum('kr,hprd->hpkd',haar,c)/4).to(torch.bfloat16)
        return pk.contiguous(),base.contiguous(),detail.contiguous()

class AllSparse(nn.Module):
    def __init__(self,rt):super().__init__();self.layers=nn.ModuleDict({str(l):SparseLayer(rt,l) for l in range(1,30)})
    def __getitem__(self,l):return self.layers[str(l)]
    @torch.no_grad()
    def project(self):
        for m in self.layers.values():
            m.pool_k.clamp_(-3,3);m.pool_v.clamp_(-.5,.5);m.pool_feature.clamp_(-.25,.25);m.route.clamp_(-3,3);m.comp_logscale.clamp_(-.25,.25);m.group_gate.clamp_(-2,2)
