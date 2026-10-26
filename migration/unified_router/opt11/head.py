"""Exact deployment logits in forward; factorized FP16/tanh surrogate gradients."""
import math
import torch
import torch.nn.functional as F
import triton as tr
from ..learning import ActionHead
from ..opt6.factor import _factor_prepare
from ..fast_route import action_dict
from .logits import _logits

def actions(d):
 h,n,b=d['score'].shape
 a=action_dict(d['rem'].reshape(-1,8),d['add'].reshape(-1,8),d['base_mask'])
 return {k:v.reshape(h,n,81)for k,v in a.items()}

def apply(d,choice):
 a=actions(d);i=a['remove'].gather(-1,choice[...,None]);j=a['add'].gather(-1,choice[...,None]);page=torch.arange(d['score'].shape[-1],device=choice.device)
 return (d['base_mask']&(page!=i))|(page==j)

def subset(d,indices):
 keys=['stats','cv','score','base_mask','rem','add']
 return {k:v[:,indices].contiguous()if k in keys else v for k,v in d.items()}

@torch.no_grad()
def exact(head,d):
 h,n,b=d['score'].shape;tmp=torch.empty((h*n,384),device=d['score'].device,dtype=torch.float16);out=torch.empty((h,n,81),device=tmp.device,dtype=torch.float32)
 st=torch.tensor(float(d['step']),device=tmp.device)
 _factor_prepare[(h*n,)](d['stats'],d['score'],d['cv'],d['costs'],d['origins'],d['rem'],d['add'],head.w1,head.b1,head.mean,head.std,st,tmp,n,b,d['history_tokens'],4680,d['beta'],'fp16',num_warps=1,enable_fp_fusion=False)
 _logits[(h*n,)](tmp,d['stats'],d['base_mask'],d['rem'],d['add'],head.w1,head.w2,head.b2,out,n,b,tr.next_power_of_2(b),True,num_warps=1,enable_fp_fusion=False)
 return out

def surrogate(head,d):
 st=d['stats'];score=d['score'];h,n,b=score.shape;center=st[...,32:64];av=center-st[...,:32];mean=st[...,66:67];std=st[...,67:68];ri=d['rem'].long();aj=d['add'].long();c=d['costs'];o=d['origins']
 def side(ix,cv):
  valid=ix>=0;diff=torch.where(valid[...,None],cv-center[:,:,None,:],0.);z=torch.where(valid,(score.gather(-1,ix.clamp_min(0))-mean)/std,0.)
  raw=torch.stack((diff.square().sum(-1),(diff*av[:,:,None]).sum(-1),z,c[ix.clamp_min(0)]/64*valid,valid.float(),o[ix.clamp_min(0)]*valid),-1)
  return torch.cat((raw,torch.zeros((h,n,1,6),device=raw.device)),-2),diff
 fi,xi=side(ri,d['cv'][...,:8,:]);fj,xj=side(aj,d['cv'][...,8:,:]);shared=torch.zeros((h,n,24),device=st.device)
 for k,v in {0:av.square().sum(-1),6:st[...,64],7:st[...,65],10:std[...,0].log(),11:d['beta'],12:st[...,68]/st[...,69],13:st[...,70]/4680,14:math.log1p(d['history_tokens']),15:math.log1p(4680),16:d['step']/3,23:st[...,72]}.items():shared[...,k]=v
 common=torch.zeros(24,device=st.device,dtype=torch.bool);common[[0,6,7,10,11,12,13,14,15,16,23]]=True
 norm=torch.where(common,(shared-head.mean[:,None])/head.std[:,None],0.).half().float();w=head.w1.half().float()
 base=((norm[...,None,:]*w[:,None]).sum(-1)+head.b1[:,None]).half().float()
 ix=torch.tensor([1,3,8,17,19,21],device=st.device);jx=ix+1
 fi=((fi-head.mean[:,None,None,ix])/head.std[:,None,None,ix]).half();fj=((fj-head.mean[:,None,None,jx])/head.std[:,None,None,jx]).half()
 yi=torch.bmm(fi.reshape(h,n*9,6),head.w1[:,:,ix].half().transpose(1,2)).reshape(h,n,9,16).float()
 yj=torch.bmm(fj.reshape(h,n*9,6),head.w1[:,:,jx].half().transpose(1,2)).reshape(h,n,9,16).float()
 cross=torch.matmul(xi.half().float(),xj.half().float().transpose(-1,-2));cross=((cross-head.mean[:,None,None,None,5])/head.std[:,None,None,None,5]).half().float()
 zero=(-head.mean[:,5]/head.std[:,5]).half().float()[:,None,None].expand(h,n,17);cross=torch.cat((zero,cross.flatten(-2)),-1)
 a=torch.arange(81,device=st.device);si=torch.where(a<17,a-9,(a-17)//8);sj=torch.where(a<9,a-1,(a-17)%8);si=torch.where(a>=9,si,8);sj=torch.where(((a>=1)&(a<=8))|(a>=17),sj,8)
 y=base[:,:,None]+yi[:,:,si]+yj[:,:,sj]+cross[...,None]*w[:,None,None,:,5]
 pred=(F.gelu(y,approximate='tanh')*head.w2[:,None]).sum(-1)+head.b2[:,None]
 return torch.cat((torch.zeros_like(pred[...,:1]),pred[...,1:]),-1)

class AlignedHead(ActionHead):
 def forward(self,d):
  hard=exact(self,d)
  if not torch.is_grad_enabled():return hard
  approx=surrogate(self,d);valid=actions(d)['valid'];approx=torch.where(valid,approx,0.);safe=torch.where(valid,hard,0.)
  return (approx+(safe-approx).detach()).masked_fill(~valid,-torch.inf)
