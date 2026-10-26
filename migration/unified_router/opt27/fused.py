"""Fused refinement experiments: retain query/candidate scores on chip."""
import math
import torch,triton as tr
import triton.language as tl
from ..opt17.route import _chosen
from ..opt26.radix import _radix

@tr.jit
def _score(Q,K,LC,IDS,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,G:tl.constexpr,GP:tl.constexpr,P:tl.constexpr,M:tl.constexpr,g,h,base,half):
 ci=base+tl.arange(0,GP);t=tl.arange(0,GP*P);d=tl.arange(0,64);r=tl.arange(0,M);qi=g*32+half*M+r
 page=tl.load(IDS+(h*G+g)*C+base+t//P,base+t//P<C,0)
 q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0)
 k=tl.load(K+((h*B+page[None,:])*16+t[None,:]%P)*64+d[:,None])
 ll=tl.dot(q,k).reshape((M,GP,P))*0.08838834764831845
 pc=tl.load(IDS+(h*G+g)*C+ci,ci<C,0);lc=tl.load(LC+(h*B+pc[:,None])*16+tl.arange(0,P)[None,:])
 ll=ll+lc[None,:,:];mx=tl.max(ll,2);score=mx+tl.log(tl.sum(tl.exp(ll-mx[:,:,None]),2))
 return tl.where(ci[None,:]<C,score,float('-inf'))

@tr.jit
def _finish(score,IDS,OUT,PART,N:tl.constexpr,C:tl.constexpr,G:tl.constexpr,KEEP:tl.constexpr,CC:tl.constexpr,M:tl.constexpr,g,h,half):
 c=tl.arange(0,CC);r=tl.arange(0,M);qi=g*32+half*M+r
 score=tl.where(c[None,:]<C,score,float('-inf'));mx=tl.max(score,1);mx=tl.where(qi<N,mx,0.)
 prob=tl.exp(score-mx[:,None]);prob=prob/tl.maximum(tl.sum(prob,1),1e-20)[:,None];prob=tl.where(qi[:,None]<N,prob,0.)
 imp=tl.sum(prob,0)
 if M==32:
  imp=tl.where(c<C,imp,float('-inf'))
  if C>512:chosen=_radix(imp,c,C,KEEP,8)&(c<C)
  else:chosen=_chosen(imp,c,C,KEEP)&(c<C)
  page=tl.load(IDS+(h*G+g)*C+c,c<C,0);rank=tl.cumsum(chosen.to(tl.int32),0)-1;tl.store(OUT+(h*G+g)*KEEP+rank,page,chosen)
 else:tl.store(PART+((h*G+g)*2+half)*C+c,imp,c<C)

@tr.jit
def _candidate_loop(Q,K,LC,IDS,OUT,PART,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,G:tl.constexpr,PAGES:tl.constexpr,KEEP:tl.constexpr,CC:tl.constexpr,GP:tl.constexpr,M:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);half=tl.program_id(2);slots=tl.arange(0,CC//GP)
 full=tl.full((M,CC//GP,GP),float('-inf'),tl.float32)
 for block in range(tr.cdiv(C,GP)):
  ci=block*GP+tl.arange(0,GP);pc=tl.load(IDS+(h*G+g)*C+ci,ci<C,0);age=tr.cdiv(B,PAGES)-1-pc//PAGES
  np=tl.where(age==0,16,tl.where(age==1,8,tl.where(age==2,4,2)));mp=tl.max(tl.where(ci<C,np,0),0)
  if mp<=2:score=_score(Q,K,LC,IDS,N,B,C,G,GP,2,M,g,h,block*GP,half)
  elif mp<=4:score=_score(Q,K,LC,IDS,N,B,C,G,GP,4,M,g,h,block*GP,half)
  elif mp<=8:score=_score(Q,K,LC,IDS,N,B,C,G,GP,8,M,g,h,block*GP,half)
  else:score=_score(Q,K,LC,IDS,N,B,C,G,GP,16,M,g,h,block*GP,half)
  full=tl.where(slots[None,:,None]==block,score[:,None,:],full)
 _finish(full.reshape((M,CC)),IDS,OUT,PART,N,C,G,KEEP,CC,M,g,h,half)

@tr.jit
def _prototype_loop(Q,K,LC,IDS,OUT,PART,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,G:tl.constexpr,KEEP:tl.constexpr,CC:tl.constexpr,PS:tl.constexpr,M:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);half=tl.program_id(2);c=tl.arange(0,CC);r=tl.arange(0,M);d=tl.arange(0,64);t=tl.arange(0,CC*PS)
 qi=g*32+half*M+r;pc=tl.load(IDS+(h*G+g)*C+c,c<C,0);page=tl.load(IDS+(h*G+g)*C+t//PS,t//PS<C,0)
 q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0)
 mx=tl.full((M,CC),float('-inf'),tl.float32);den=tl.zeros((M,CC),tl.float32)
 for chunk in range(16//PS):
  pp=chunk*PS+t%PS
  k=tl.load(K+((h*B+page[None,:])*16+pp[None,:])*64+d[:,None])
  ll=tl.dot(q,k).reshape((M,CC,PS))*0.08838834764831845
  lc=tl.load(LC+(h*B+pc[:,None])*16+chunk*PS+tl.arange(0,PS)[None,:]);ll=ll+lc[None,:,:]
  mm=tl.maximum(mx,tl.max(ll,2));safe=tl.where(mm==float('-inf'),0.,mm)
  den=den*tl.exp(mx-safe)+tl.sum(tl.exp(ll-safe[:,:,None]),2);mx=mm
 score=mx+tl.log(den)
 _finish(score,IDS,OUT,PART,N,C,G,KEEP,CC,M,g,h,half)

@tr.jit
def _mass_loop(Q,K,LC,IDS,OUT,PART,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,G:tl.constexpr,KEEP:tl.constexpr,CC:tl.constexpr,PS:tl.constexpr,M:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);half=tl.program_id(2);c=tl.arange(0,CC);r=tl.arange(0,M);d=tl.arange(0,64);t=tl.arange(0,CC*PS)
 qi=g*32+half*M+r;pc=tl.load(IDS+(h*G+g)*C+c,c<C,0);page=tl.load(IDS+(h*G+g)*C+t//PS,t//PS<C,0)
 q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0)
 mx=tl.full((M,),float('-inf'),tl.float32);den=tl.zeros((M,CC),tl.float32)
 for chunk in range(16//PS):
  pp=chunk*PS+t%PS
  k=tl.load(K+((h*B+page[None,:])*16+pp[None,:])*64+d[:,None])
  ll=tl.dot(q,k).reshape((M,CC,PS))*0.08838834764831845
  lc=tl.load(LC+(h*B+pc[:,None])*16+chunk*PS+tl.arange(0,PS)[None,:]);ll=ll+lc[None,:,:]
  ll=tl.where(c[None,:,None]<C,ll,float('-inf'))
  mm=tl.maximum(mx,tl.max(tl.max(ll,2),1));safe=tl.where(mm==float('-inf'),0.,mm)
  den=den*tl.exp(mx-safe)[:,None]+tl.sum(tl.exp(ll-safe[:,None,None]),2);mx=mm
 prob=den/tl.maximum(tl.sum(den,1),1e-20)[:,None];prob=tl.where(qi[:,None]<N,prob,0.)
 imp=tl.sum(prob,0)
 if M==32:
  imp=tl.where(c<C,imp,float('-inf'))
  if C>512:chosen=_radix(imp,c,C,KEEP,8)&(c<C)
  else:chosen=_chosen(imp,c,C,KEEP)&(c<C)
  page=tl.load(IDS+(h*G+g)*C+c,c<C,0);rank=tl.cumsum(chosen.to(tl.int32),0)-1;tl.store(OUT+(h*G+g)*KEEP+rank,page,chosen)
 else:tl.store(PART+((h*G+g)*2+half)*C+c,imp,c<C)


@tr.jit
def _merge(PART,IDS,OUT,C:tl.constexpr,G:tl.constexpr,KEEP:tl.constexpr,CC:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);c=tl.arange(0,CC)
 a=tl.load(PART+((h*G+g)*2)*C+c,c<C,0);b=tl.load(PART+((h*G+g)*2+1)*C+c,c<C,0);imp=tl.where(c<C,a+b,float('-inf'))
 if C>512:chosen=_radix(imp,c,C,KEEP,8)&(c<C)
 else:chosen=_chosen(imp,c,C,KEEP)&(c<C)
 page=tl.load(IDS+(h*G+g)*C+c,c<C,0);rank=tl.cumsum(chosen.to(tl.int32),0)-1;tl.store(OUT+(h*G+g)*KEEP+rank,page,chosen)

def refine(qq,state,candidates,seg,kind='candidate',block=16,m=32,warps=4):
 h,n,_=qq.shape;_,g,c=candidates.shape;b=state['key'].shape[1];keep=math.ceil(b*.2);cc=tr.next_power_of_2(c)
 out=torch.empty((h,g,keep),device=qq.device,dtype=torch.int32);part=torch.empty((h,g,2,c),device=qq.device) if m==16 else out
 if kind=='candidate':_candidate_loop[(g,h,32//m)](qq,state['key'],state['logcount'],candidates,out,part,n,b,c,g,tr.cdiv(seg,64),keep,cc,block,m,num_warps=warps,num_stages=1)
 elif kind=='mass':_mass_loop[(g,h,32//m)](qq,state['key'],state['logcount'],candidates,out,part,n,b,c,g,keep,cc,block,m,num_warps=warps,num_stages=1)
 else:_prototype_loop[(g,h,32//m)](qq,state['key'],state['logcount'],candidates,out,part,n,b,c,g,keep,cc,block,m,num_warps=warps,num_stages=1)
 if m==16:_merge[(g,h)](part,candidates,out,c,g,keep,cc,num_warps=4)
 return out
