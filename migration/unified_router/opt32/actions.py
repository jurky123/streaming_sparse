"""Shared physical-tile actions. Online inputs: proxy masses and K/V page means.
Original dense page statistics exist only in counterfactual training labels.
"""
import math
import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl
from ..opt30 import kernels
from ..opt28.pipeline import query
from ..opt16.grouping import order,restore
from ..opt30.model import page_statistics

def context(pipe,model,q,raw):
 o=model.options;kv=pipe.current(raw['kc'],raw['vc']);ix=order(q,model.tree('q'));qq,qi,qs=query(q,ix,kv.state['basis'],o['g'])
 if o['mode']=='local':bar,mass=kernels.local_aggregate(qq,model.query_weight,o['g'],o['r'])
 else:bar,mass=kernels.aggregate(qq,ix,None,o['g'],o['r'],o['mode'])
 ids,d=kernels.route(qq,kv.state,bar,mass,pipe.seg,o['g'],o['k'],o['candidate']);p=d['importance'];mask=torch.zeros_like(p,dtype=torch.bool).scatter_(-1,ids.long(),True)
 return dict(kv=kv,ix=ix,qi=qi,qs=qs,p=p,mask=mask,ids=ids,seg=pipe.seg,g=o['g'],k=o['k'],q_dtype=q.dtype)

def candidates(d):
 p,mask=d['p'],d['mask'];h,t,b=p.shape;pages=math.ceil(d['seg']/d['k']);page=torch.arange(b,device=p.device);full=(page%pages*d['k']+d['k']<=d['seg']);row=torch.arange(h*t,device=p.device).reshape(h,t,1)
 def side(pool,score):
  first=score.masked_fill(~pool,torch.inf).argsort(dim=-1,stable=True)[...,:6];good=pool.gather(-1,first);remaining=pool.clone().scatter_(-1,first,False)
  hashed=((page*1103515245+row*12345)%2147483647).float();second=hashed.masked_fill(~remaining,torch.inf).argsort(dim=-1,stable=True)[...,:2];ok=remaining.gather(-1,second)
  return torch.cat((first.masked_fill(~good,-1),second.masked_fill(~ok,-1)),-1)
 ri=side(mask&full,p);aj=side(~mask&full&(p>0),-p);absent=torch.full((h,t,1),-1,device=p.device,dtype=torch.long)
 i=torch.cat((absent,absent.expand(-1,-1,8),ri,ri.repeat_interleave(8,-1)),-1);j=torch.cat((absent,aj,absent.expand(-1,-1,8),aj.repeat(1,1,8)),-1)
 index=torch.arange(81,device=p.device);types=(index>0).long()+(index>8).long()+(index>16).long()
 valid=torch.cat((torch.ones_like(absent,dtype=torch.bool),aj>=0,(ri>=0)&(mask.sum(-1,keepdim=True)>1),(ri.repeat_interleave(8,-1)>=0)&(aj.repeat(1,1,8)>=0)),-1)
 return dict(i=i,j=j,types=types,valid=valid,cost=(j>=0).float()-(i>=0).float())

def content(d,model,kind,basis):
 x=d['kv'].k if kind=='K32'else d['kv'].v;h,n,dim=x.shape;seg=d['seg'];k=d['k'];pages=math.ceil(seg/k)
 xx=F.pad(x.float().reshape(h,n//seg,seg,dim),(0,0,0,pages*k-seg)).reshape(h,-1,k,dim)
 counts=(seg-torch.arange(pages,device=x.device)*k).clamp(max=k).repeat(n//seg)
 means=xx.sum(2)/counts[None,:,None]
 projection=model.routing['key_projection'][...,:32]if kind=='K32'else basis[...,:int(kind[1:])]
 return torch.bmm(means,projection.float())

def features(d,a,c):
 p,mask=d['p'],d['mask'];z=(p*mask).sum(-1).clamp_min(1e-8);center=torch.bmm(p*mask,c)/z[...,None];total=torch.bmm(p,c);error=center-total
 def mass(ix):return p.gather(-1,ix.clamp_min(0))*(ix>=0)
 def vector(ix):
  h,t,n=ix.shape;r=c.shape[-1];v=c.gather(1,ix.clamp_min(0).flatten(1)[...,None].expand(-1,-1,r)).reshape(h,t,n,r)
  return (v-center[:,:,None])*(ix>=0)[...,None]
 i,j=a['i'],a['j'];mi,mj=mass(i),mass(j);vi,vj=vector(i),vector(j);den=(z[...,None]-mi+mj).clamp_min(1e-8)
 delta=(mj[...,None]*vj-mi[...,None]*vi)/den[...,None];energy=(total.square().sum(-1)+torch.bmm(p,c.square().sum(-1,keepdim=True))[...,0]).clamp_min(1e-6)
 norm=lambda x:x/energy[...,None]
 linear=norm(-2*(error[:,:,None]*delta).sum(-1));quadratic=norm(delta.square().sum(-1));gain=linear-quadratic
 global_var=(torch.bmm(p,c.square().sum(-1,keepdim=True))[...,0]-total.square().sum(-1)).clamp_min(0)/energy
 selected_var=(torch.bmm(p*mask,c.square().sum(-1,keepdim=True))[...,0]/z-center.square().sum(-1)).clamp_min(0)/energy
 constant=lambda x:x[...,None].expand_as(mi)
 pages=math.ceil(d['seg']/d['k']);b=p.shape[-1];ci=(i>=b-pages)&(i>=0);cj=(j>=b-pages)&(j>=0)
 x=torch.stack((gain,linear,quadratic,(den/z[...,None]).log(),mi/z[...,None],mj/z[...,None],norm(vi.square().sum(-1)),norm(vj.square().sum(-1)),norm((vi*vj).sum(-1)),constant(error.square().sum(-1)/energy),constant(global_var),constant(selected_var),mi,mj,(mj+1e-6).log()-(mi+1e-6).log(),(i>=0).float(),(j>=0).float(),(a['types']==3).float().expand_as(mi),(a['types']==1).float().expand_as(mi),(a['types']==2).float().expand_as(mi),ci.float(),cj.float(),a['cost'],constant(mask.float().mean(-1))),-1)
 return x,gain

def valid_actions(a,mode):
 valid=a['valid']
 if mode=='swap':valid=valid&((a['types']==0)|(a['types']==3))
 if mode=='none':valid=valid&(a['types']==0)
 return valid

def apply(d,a,chosen):
 i=a['i'].gather(-1,chosen[...,None]);j=a['j'].gather(-1,chosen[...,None]);page=torch.arange(d['mask'].shape[-1],device=chosen.device)
 return(d['mask']&(page!=i))|(page==j)

@tr.jit
def _maximum(a,b):
 return tl.maximum(a,b)

@tr.jit
def _mask_metadata(M,L,C,T:tl.constexpr,B:tl.constexpr,BB:tl.constexpr):
 row=tl.program_id(0);page=tl.arange(0,BB);m=tl.load(M+row*B+page,page<B,False)
 rank=tl.cumsum(m.to(tl.int32),0)-1;count=tl.sum(m.to(tl.int32),0)
 prior=tl.load(M+row*B+page-1,(page>0)&(page<=B),False)
 prev=tl.associative_scan(tl.where(prior,page-1,0),0,_maximum)
 tl.store(L+row*B+rank,page-prev,m);tl.store(C+row,count)

def attend(pipe,d,mask):
 h,t,b=mask.shape;lut=torch.empty((h,t,b),device=mask.device,dtype=torch.int32);counts=torch.empty((h,t),device=mask.device,dtype=torch.int32)
 _mask_metadata[(h*t,)](mask,lut,counts,t,b,tr.next_power_of_2(b),num_warps=4)
 lib=pipe.lib;out=torch.empty(d['qi'].shape,device=mask.device,dtype=d['q_dtype']);lib.module.run(d['qi'],lib.ki,lib.v,out,lut,counts,d['qs'],lib.ks,lib.g,lib.k,lib.w,lib.pages,lib.seg%lib.k or lib.k)
 return restore(out,d['ix'])

@torch.no_grad()
def labels(q,targets,d,a):
 h,n,_=q.shape;inv=torch.empty_like(d['ix']);inv.scatter_(1,d['ix'].long(),torch.arange(n,device=q.device,dtype=d['ix'].dtype)[None].expand(h,-1));group=inv[:,targets].long()//d['g'];head=torch.arange(h,device=q.device)[:,None]
 logz,value=page_statistics(q[:,targets],d['kv'],d['seg'],d['k']);p=logz.softmax(-1);truth=torch.einsum('hnb,hnbd->hnd',p,value);mask=d['mask'][head,group];z=(p*mask).sum(-1).clamp_min(1e-20);base=torch.einsum('hnb,hnbd->hnd',p*mask,value)/z[...,None]
 i,j=a['i'][head,group],a['j'][head,group];page=torch.arange(p.shape[-1],device=q.device)
 masks=(mask[:,:,None,:]&(page!=i[...,None]))|(page==j[...,None]);masks=torch.where(a['valid'][head,group][...,None],masks,mask[:,:,None,:])
 # Renormalize each actual hard action in log-space; subtracting a dominant
 # removed page's probability from the base denominator is numerically unsafe.
 weights=logz[:,:,None,:].masked_fill(~masks,-torch.inf).softmax(-1)
 outputs=torch.einsum('hnab,hnbd->hnad',weights,value);delta=outputs-base[:,:,None]
 errors=(outputs-truth[:,:,None]).square().sum(-1)
 return dict(group=group,head=head,errors=errors,truth=truth,delta=delta,base_error_vector=base-truth)
