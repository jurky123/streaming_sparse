"""Reuse coarse prototypes across query groups; contiguous tile fine scores."""
import math
import torch
import triton as tr
import triton.language as tl
from ..opt8.projection import project
from ..opt17.route import _top,_chosen
from ..opt23.route import _fine_age as original_fine
from ..opt19.tile import select as original_select
from .radix import _radix,top as radix_top

@tr.jit
def _coarse_grouped(Q,K,LC,S,N:tl.constexpr,B:tl.constexpr,G:tl.constexpr,OFF:tl.constexpr,NP:tl.constexpr,P:tl.constexpr,GP:tl.constexpr,MG:tl.constexpr):
 h=tl.program_id(2);gm=tl.program_id(0)*MG
 m=tl.arange(0,MG*8);d=tl.arange(0,64);pp=tl.arange(0,GP*P);pg=tl.program_id(1)*GP+tl.arange(0,GP)
 qi=tl.minimum((gm+m//8)*32+(m%8)*4+1,N-1)
 q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:])
 pages=tl.program_id(1)*GP+pp//P
 k=tl.load(K+(h*NP*P+pages[None,:]*P+pp[None,:]%P)*64+d[:,None],pages[None,:]<NP,0)
 logits=tl.dot(q,k).reshape((MG*8,GP,P))*0.08838834764831845
 lc=tl.load(LC+(h*NP+pg[:,None])*P+tl.arange(0,P)[None,:],pg[:,None]<NP,float('-inf'))
 logits=logits+lc[None,:,:];mx=tl.max(logits,2);safe=tl.where(mx==float('-inf'),0.,mx)
 score=safe+tl.log(tl.sum(tl.exp(logits-safe[:,:,None]),2))
 score=tl.max(score.reshape((MG,8,GP)),1)
 gg=gm+tl.arange(0,MG)
 tl.store(S+(h*G+gg[:,None])*B+OFF+pg[None,:],score,(gg[:,None]<G)&(pg[None,:]<NP))

@tr.jit
def _part(Q,K,LC,IDS,S,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,G:tl.constexpr,GP:tl.constexpr,P:tl.constexpr,TRANS:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(2);ci=tl.program_id(1)*GP+tl.arange(0,GP)
 t=tl.arange(0,GP*P);d=tl.arange(0,64);qi=g*32+tl.arange(0,32)
 ix=tl.program_id(1)*GP+t//P;page=tl.load(IDS+(h*G+g)*C+ix,ix<C,0)
 q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0)
 k=tl.load(K+((h*B+page[None,:])*16+t[None,:]%P)*64+d[:,None])
 ll=tl.dot(q,k).reshape((32,GP,P))*0.08838834764831845
 pc=tl.load(IDS+(h*G+g)*C+ci,ci<C,0);lc=tl.load(LC+(h*B+pc[:,None])*16+tl.arange(0,P)[None,:])
 ll=ll+lc[None,:,:];mx=tl.max(ll,2);score=mx+tl.log(tl.sum(tl.exp(ll-mx[:,:,None]),2))
 if TRANS:offset=((h*G+g)*C+ci[None,:])*32+tl.arange(0,32)[:,None]
 else:offset=(h*N+qi[:,None])*C+ci[None,:]
 tl.store(S+offset,score,(qi[:,None]<N)&(ci[None,:]<C))

@tr.jit
def _fine(Q,K,LC,IDS,S,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,G:tl.constexpr,PAGES:tl.constexpr,GP:tl.constexpr,TRANS:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(2);ci=tl.program_id(1)*GP+tl.arange(0,GP);pc=tl.load(IDS+(h*G+g)*C+ci,ci<C,0)
 age=tr.cdiv(B,PAGES)-1-pc//PAGES;np=tl.where(age==0,16,tl.where(age==1,8,tl.where(age==2,4,2)));mp=tl.max(tl.where(ci<C,np,0),0)
 if mp<=2:_part(Q,K,LC,IDS,S,N,B,C,G,GP,2,TRANS)
 elif mp<=4:_part(Q,K,LC,IDS,S,N,B,C,G,GP,4,TRANS)
 elif mp<=8:_part(Q,K,LC,IDS,S,N,B,C,G,GP,8,TRANS)
 else:_part(Q,K,LC,IDS,S,N,B,C,G,GP,16,TRANS)

@tr.jit
def _tiles(S,IDS,OUT,N:tl.constexpr,C:tl.constexpr,G:tl.constexpr,KEEP:tl.constexpr,CC:tl.constexpr,TRANS:tl.constexpr,RADIX:tl.constexpr):
 tile=tl.program_id(0);h=tl.program_id(1);r=tl.arange(0,32);q=tile*32+r;c=tl.arange(0,CC)
 if TRANS:off=((h*G+tile)*C+c[None,:])*32+r[:,None]
 else:off=(h*N+q[:,None])*C+c[None,:]
 score=tl.load(S+off,(q[:,None]<N)&(c[None,:]<C),float('-inf'))
 mx=tl.max(score,1);mx=tl.where(q<N,mx,0.);prob=tl.exp(score-mx[:,None]);prob=prob/tl.maximum(tl.sum(prob,1),1e-20)[:,None]
 importance=tl.sum(prob,0);importance=tl.where(c<C,importance,float('-inf'))
 if RADIX:chosen=_radix(importance,c,C,KEEP,8)&(c<C)
 else:chosen=_chosen(importance,c,C,KEEP)&(c<C)
 page=tl.load(IDS+(h*G+tile)*C+c,c<C,0);rank=tl.cumsum(chosen.to(tl.int32),0)-1;tl.store(OUT+(h*G+tile)*KEEP+rank,page,chosen)

def coarse(qq,state,mg=8,width=128,warps=4,stages=1):
 h,n,_=qq.shape;g=tr.cdiv(n,32);b=state['key'].shape[1];out=torch.empty((h,g,b),device=qq.device)
 off=0
 for kk,lc in state['parts']:
  np,p=lc.shape[1:];gp=width//p
  _coarse_grouped[(tr.cdiv(g,mg),tr.cdiv(np,gp),h)](qq,kk,lc,out,n,b,g,off,np,p,gp,mg,num_warps=warps,num_stages=stages)
  off+=np
 return out

def refine(qq,state,candidates,seg,gp=8,warps=4,trans=False):
 h,n,_=qq.shape;g=tr.cdiv(n,32);b=state['key'].shape[1];c=candidates.shape[-1]
 shape=(h,g,c,32) if trans else (h,n,c);out=torch.empty(shape,device=qq.device)
 _fine[(g,tr.cdiv(c,gp),h)](qq,state['key'],state['logcount'],candidates,out,n,b,c,g,tr.cdiv(seg,64),gp,trans,num_warps=warps,num_stages=1)
 return out

def select(fine,candidates,n,b,trans=False,warps=4,radix=False):
 h,g,c=candidates.shape;keep=math.ceil(b*.2);out=torch.empty((h,g,keep),device=fine.device,dtype=torch.int32)
 _tiles[(g,h)](fine,candidates,out,n,c,g,keep,tr.next_power_of_2(c),trans,radix,num_warps=warps)
 return out

def route(q,state,seg,mg=8,width=128,coarse_warps=4,gp=8,fine_warps=4,trans=False,radix=False,tile_radix=None,legacy_fine=False):
 h,n,_=q.shape;g=tr.cdiv(n,32);b=state['key'].shape[1];c=math.ceil(b*.35);qq=project(q,state['basis'])
 score=coarse(qq,state,mg,width,coarse_warps)
 if radix:candidates=radix_top(score,c,8,4)
 else:
  candidates=torch.empty((h,g,c),device=q.device,dtype=torch.int32)
  _top[(h*g,)](score,candidates,h*g,b,c,tr.next_power_of_2(b),num_warps=4)
 if legacy_fine:
  fine=torch.empty((h,n,c),device=q.device)
  original_fine[(g,tr.cdiv(c,8),h)](qq,state['key'],state['logcount'],candidates,fine,n,b,c,g,tr.cdiv(seg,64),num_warps=4,num_stages=1)
 else:fine=refine(qq,state,candidates,seg,gp,fine_warps,trans)
 use_radix=radix if tile_radix is None else tile_radix
 if legacy_fine and not use_radix:return original_select(fine,candidates,b,32,.2)
 return select(fine,candidates,n,b,trans,radix=use_radix)


def optimized(q,state,seg):
 b=state['key'].shape[1];c=math.ceil(b*.35)
 return route(q,state,seg,mg=8,width=128 if b>1024 else 64,radix=b>512,tile_radix=c>512,legacy_fine=True)
