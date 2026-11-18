import math
import triton as tr
import triton.language as tl
from local_experiments.review_v1.tuned_attention import _av,_attn_fwd
@tr.jit
def row_tail(Q,KEY,MEAN,DETAIL,MASK,MU,CODE,LSE,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=64,BR:tl.constexpr=64, BF16_AV:tl.constexpr=False,MERGE:tl.constexpr=False,EXACT=None,ELSE=None,BASIS=None,OUT=None,HEAD_PAGES:tl.constexpr=0,K=None,V=None,IDS=None,EXACT_COUNT:tl.constexpr=0,QK_SCALE:tl.constexpr=0.,ANCHOR_SCALE:tl.constexpr=False,COLUMN_MASS:tl.constexpr=False):
 hp:tl.constexpr=HEAD_PAGES if HEAD_PAGES else P
 t=tl.program_id(0);h=tl.program_id(1);qi=t*BQ+tl.arange(0,BQ);d=tl.arange(0,D);r=tl.arange(0,BR);pages=tl.arange(0,BR//4);rank=tl.arange(0,16)
 q=tl.load(Q+(qi[:,None]*H+h)*D+d[None,:])
 groups:tl.constexpr=BQ//64
 group=tl.arange(0,groups)
 lut=IDS+(h*(N//64)+t*groups+group[:,None])*EXACT_COUNT
 mx=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32)
 am=tl.zeros((BQ,D),tl.float32);ac=tl.zeros((BQ,16),tl.float32)
 if EXACT_COUNT:
  slots=tl.arange(0,tr.next_power_of_2(EXACT_COUNT))
  ids=tl.load(lut+slots[None,:],slots[None,:]<EXACT_COUNT,0)
  bit=tl.where(slots[None,:]<EXACT_COUNT,1<<(ids%32),0).to(tl.uint32)
  words=tl.sum(tl.where(tl.arange(0,16)[None,:,None]==ids[:,None,:]//32,bit[:,None,:],0),2)
 for first in range(tr.cdiv(P*4,BR)):
  rep=first*BR+r;page=first*(BR//4)+pages
  word=tl.sum(tl.where(tl.arange(0,16)[None,:]==first//2,words,0),1)
  omitted=((word[:,None]>>(rep[None,:]//4%32))&1)==0
  keep=tl.reshape(tl.broadcast_to(omitted[:,None,:],(groups,64,BR)),(BQ,BR))
  key=tl.load(KEY+(h*hp*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
  z=tl.dot(q,key)*(1.4426950408889634/math.sqrt(D))
  z=tl.where((rep[None,:]<P*4)&keep,z,-float('inf'))
  nm=tl.maximum(mx,tl.max(z,1));fade=tl.exp2(mx-nm);w=tl.exp2(z-nm[:,None]);ww=tl.reshape(w,(BQ,BR//4,4));rr=tl.arange(0,4)
  alpha=tl.sum(ww,2)
  c0=tl.sum(ww*tl.where(rr<2,1.,-1.)[None,None,:],2)
  c1=tl.sum(ww*tl.where(rr%2==0,1.,-1.)[None,None,:],2)
  c2=tl.sum(ww*tl.where((rr==0)|(rr==3),1.,-1.)[None,None,:],2)
  mean=tl.load(MEAN+(h*hp+page[:,None])*D+d[None,:],page[:,None]<P,0).to(tl.float32)
  x=tl.load(DETAIL+((h*hp+page[:,None])*3)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
  y=tl.load(DETAIL+((h*hp+page[:,None])*3+1)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
  v=tl.load(DETAIL+((h*hp+page[:,None])*3+2)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
  if BF16_AV:
   am=am*fade[:,None]+_av(alpha,mean)
   ac=ac*fade[:,None]+_av(c0,x)+_av(c1,y)+_av(c2,v)
  else:
   am=am*fade[:,None]+tl.dot(alpha,mean,input_precision='tf32x3')
   ac=ac*fade[:,None]+tl.dot(c0,x,input_precision='tf32x3')+tl.dot(c1,y,input_precision='tf32x3')+tl.dot(c2,v,input_precision='tf32x3')
  den=den*fade+tl.sum(w,1);mx=nm
 safe=tl.maximum(den,1.e-30)
 if MERGE:
  code=tl.where(rank[None,:]<8,ac/safe[:,None],0.)
  basis=tl.load(BASIS+(h*D+d[None,:])*8+rank[:,None],rank[:,None]<8,0).to(tl.float32)
  tail=am/safe[:,None]+tl.dot(code,basis,input_precision='tf32x3')
  le=tl.load(ELSE+qi*H+h)
  exact=tl.load(EXACT+(qi[:,None]*H+h)*D+d[None,:])
  lt=tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf'))
  rho=1./(1.+tl.exp2(le-lt))
  tl.store(OUT+(qi[:,None]*H+h)*D+d[None,:],exact*(1.-rho[:,None])+tail*rho[:,None])
 else:
  tl.store(MU+(h*N+qi[:,None])*D+d[None,:],am/safe[:,None])
  tl.store(CODE+(h*N+qi[:,None])*8+rank[None,:],ac/safe[:,None],rank[None,:]<8)
  tl.store(LSE+h*N+qi,tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf')))

@tr.jit
def guard(Q,K,V,IDS,O,L,M,P:tl.constexpr,COUNT:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);qi=t*64+tl.arange(0,64)
 row=qi*12+h
 mass=tl.load(M+row)
 if tl.min(((mass>=1.)&(mass<1.21e24)).to(tl.int32),0)==0:
  e,le=_attn_fwd(Q,K,V,128**-.5,COUNT,IDS,O,O,12,4608,P*64,72,128,64,64,HIGH_PRECISION_AV=True,RETURN_VALUES=True)
  tl.store(O+row[:,None]*128+tl.arange(0,128)[None,:],e);tl.store(L+row,le)
