"""Frozen models split into query-independent page encodings and query readouts."""
import torch
from ..review_v1.ops import Summary

@torch.no_grad()
def encode(summary,reader,phi,B):
    h,p=summary.mean.shape[:2]
    parts=[summary.keys.float().flatten(2),summary.mean.float(),summary.detail.float().flatten(2),summary.stats.float().flatten(2)]
    f=torch.cat([x/(x.square().mean(-1,keepdim=True).sqrt()+1e-4) for x in parts],-1)
    # These BF16 rounding points match LocalReader / AdaptivePhi exactly.
    co=reader.page(f.reshape(h*p,-1)).reshape(h,p,5,17).bfloat16().float()
    A=phi.page(f.reshape(h*p,-1)).reshape(h,p,8,4).bfloat16().float()
    return co,A,summary.mean.double()@B

@torch.no_grad()
def read(probes,co,A,phi):
    # probes H,T,M,128; page coefficients H,P,...; no encoder per tile.
    u=phi.query(probes.float());F=torch.tanh(torch.einsum('htmi,hpij->htmpj',u,A))
    val=co[:,None,None,:,0]+torch.einsum('htmpj,hpjk->htmpk',F,co[:,: ,1:])
    return val[...,:16],val[...,16]
