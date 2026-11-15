import json,math,time,csv,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..review_v1.model import Layer
from ..review_v1.ops import Summary,SummaryOps,RouteOps,Config,haar
from ..e12_swap_prediction.core import pool
from ..set_oracle_v1.common import dump,sha
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/e24_joint_representatives_v1';PRE=ROOT/'artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt';OLD=ROOT/'artifacts/set_oracle_v1'
def layer(state,l,h,device):
 vals={n:state[f'layers.{l}.{n}'][h:h+1].to(device) for n in ['bank','group_v','group_gate','pool_k','pool_v','pool_feature','basis']};m=Layer(vals)
 for n,p in m.named_parameters():p.requires_grad_(n in ['pool_k','pool_v','pool_feature'])
 return m

def summary(m,k,v,hp):
 hs=SummaryOps.build(m,k[:,:hp],v[:,:hp]);cs=SummaryOps.build(m,k[:,hp:],v[:,hp:],cheap=True)
 return Summary(*(torch.cat((getattr(hs,n),getattr(cs,n)),1) for n in ['keys','mean','detail','stats']))

def contributions(q,s,basis,scale):
 dtype=scale.dtype;keys=s.keys[0].to(dtype);nu=s.mean[0].to(dtype)[:,None]+torch.einsum('ag,pak,dk->pgd',haar(q).to(dtype),s.detail[0].to(dtype),basis[0].bfloat16().to(dtype));logit=torch.einsum('qd,pgd->qpg',q.to(dtype),keys)/math.sqrt(128);w=16*(logit-scale[:,None,None]).exp();zh=w.sum(-1);nh=torch.einsum('qpg,pgd->qpd',w,nu);A=zh.sum(1);o=nh.sum(1)/A[:,None];mass=zh/A[:,None];return dict(nh=nh,zh=zh,A=A,o=o,mass=mass,logit=logit,logzh=(logit+math.log(16)).logsumexp(-1)-scale[:,None])


def labels(c,N,Z,od,B,logZ=None):
 u=(Z-c['zh'])/c['A'][:,None];t=(N-c['nh']-c['o'][:,None]*(Z-c['zh'])[...,None])/c['A'][:,None,None];return t@B.to(t.dtype),u,(Z.log() if logZ is None else logZ)-c['logzh'],t

def mixed(c,N,Z,mask):return torch.where(mask[None,:,None],N,c['nh']).sum(1)/torch.where(mask[None],Z,c['zh']).sum(1)[:,None]
def objective(c,N,Z,od,mask):return (mixed(c,N,Z,mask)-od).square().sum()/(od.square().sum()+1e-12)
def gains(c,N,Z,od,mask,r,a):
 ns=torch.where(mask[None,:,None],N,c['nh']).sum(1);zs=torch.where(mask[None],Z,c['zh']).sum(1);dN=N-c['nh'];dZ=Z-c['zh'];base=objective(c,N,Z,od,mask);ans=[]
 for st in range(0,len(r),64):
  rr=r[st:st+64];aa=a[st:st+64];nn=ns[:,None]-dN[:,rr]+dN[:,aa];zz=zs[:,None]-dZ[:,rr]+dZ[:,aa];ans.append(base-((nn/zz[...,None]-od[:,None]).square().sum((0,2))/(od.square().sum()+1e-12)))
 return torch.cat(ans)

def route(q,s,m,own):
 with torch.no_grad():
  f,_=RouteOps.features(q[None],s);risk=(f*m.theta[:,None,None]).sum(-1).mean(1)[0];P=len(risk);mask=RouteOps.select(risk[None,None],torch.tensor([[own]],device=q.device),Config())['mask'][0,0];r,a=pool({'risk':risk.cpu().numpy()},mask.cpu().numpy(),own)
 return mask,torch.as_tensor(r,device=q.device),torch.as_tensor(a,device=q.device)
class Reader(nn.Module):
 def __init__(self):
  super().__init__();self.page=nn.Sequential(nn.Linear(712,64),nn.GELU(),nn.Linear(64,85));self.query=nn.Linear(128,4,bias=False);nn.init.normal_(self.page[-1].weight,std=.001);nn.init.zeros_(self.page[-1].bias)
 def forward(self,q,s,c):
  parts=[s.keys[0].float().flatten(1),s.mean[0].float(),s.detail[0].float().flatten(1),s.stats[0].float().flatten(1)];f=torch.cat([p/(p.detach().square().mean(-1,keepdim=True).sqrt()+1e-4) for p in parts],1);co=self.page(f).reshape(-1,5,17).to(torch.bfloat16).float();rel=c['logit']-c['logit'].mean(-1,keepdim=True);phi=torch.tanh(self.query(q.float())[:,None]/math.sqrt(128)+rel).to(co.dtype);pred=co[None,:,0]+torch.einsum('qpj,pjk->qpk',phi,co[:,1:]);z=c['mass'][...,None]*pred[...,:16];beta=pred[...,16];u=c['mass']*torch.expm1(beta.to(c['mass'].dtype));return z,u,beta

def predicted(z,u,mass,mask,r,a,beta=None):
 # Positive sum form avoids cancellation in 1+sum(u).
 exact=mass+u if beta is None else mass*beta.to(mass.dtype).exp();allD=exact.sum(1);selD=torch.where(mask[None],exact,mass).sum(1);Ta=z.sum(1);Ts=z[:,mask].sum(1);target=Ta/allD[:,None];base=(target-Ts/selD[:,None]).square().sum(-1).mean();out=[]
 for st in range(0,len(r),64):
  rr=r[st:st+64];aa=a[st:st+64];tt=Ts[:,None]-z[:,rr]+z[:,aa];dd=selD[:,None]-u[:,rr]+u[:,aa];assert (dd>0).all();out.append(base-(target[:,None]-tt/dd[...,None]).square().sum(-1).mean(0))
 return torch.cat(out)
