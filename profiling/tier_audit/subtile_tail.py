"""Latest full-mean + zero-sum detail attention; reuses repository page64 kernel.

BF16 Q/K/V and summaries, FP32 accumulation. The omitted AV supports
tf32x3 or two-term BF16 decomposition. Block sizes are explicit tuning options;
production candidate uses BQ=64, BR=64, BF16_AV=True. Numerical order differs.
"""
import math
import torch
import triton as tr
import triton.language as tl
from subtile_exact import _attn_fwd
from local_experiments.gap_exact_round_kernel import _anchored_exact
from local_experiments.high_sparse_runtime import _budget_mask

@tr.jit
def _av(a,b):
 hi=a.to(tl.bfloat16);lo=(a-hi.to(tl.float32)).to(tl.bfloat16)
 return tl.dot(hi,b.to(tl.bfloat16))+tl.dot(lo,b.to(tl.bfloat16))

@tr.jit
def _tail(Q,KEY,MEAN,DETAIL,MASK,MU,CODE,LSE,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=64,BR:tl.constexpr=64, BF16_AV:tl.constexpr=False,MERGE:tl.constexpr=False,EXACT=None,ELSE=None,BASIS=None,OUT=None,HEAD_PAGES:tl.constexpr=0,K=None,V=None,IDS=None,EXACT_COUNT:tl.constexpr=0,QK_SCALE:tl.constexpr=0.,ANCHOR_SCALE:tl.constexpr=False,COLUMN_MASS:tl.constexpr=False,PREFETCH:tl.constexpr=0):
 hp:tl.constexpr=HEAD_PAGES if HEAD_PAGES else P
 t=tl.program_id(0);h=tl.program_id(1);qi=t*BQ+tl.arange(0,BQ);d=tl.arange(0,D);r=tl.arange(0,BR);pages=tl.arange(0,BR//4);rank=tl.arange(0,16)
 q=tl.load(Q+(qi[:,None]*H+h)*D+d[None,:])
 if EXACT_COUNT:
  tl.static_assert(((BQ == 64 or BQ == 32) and BR == 64) and MERGE)
  if ANCHOR_SCALE:
   exact,le,mass=_anchored_exact(q,K,V,IDS+(h*(N//64)+t//(64//BQ))*EXACT_COUNT,h,H,D,EXACT_COUNT,QK_SCALE,COLUMN_MASS)
   # The first page contributes at least one. Recompute unsafe tiles using online softmax.
   if tl.min(((mass>=1.)&(mass<1.21e24)).to(tl.int32),0)==0:
    exact,le=_attn_fwd(Q,K,V,QK_SCALE,EXACT_COUNT,IDS,OUT,OUT,H,N,P*64,N//64,D,BQ,64,HIGH_PRECISION_AV=True,RETURN_VALUES=True,PREFETCH_PAGES=PREFETCH)
  else:
   exact,le=_attn_fwd(Q,K,V,QK_SCALE,EXACT_COUNT,IDS,OUT,OUT,H,N,P*64,N//64,D,BQ,64,HIGH_PRECISION_AV=True,RETURN_VALUES=True,PREFETCH_PAGES=PREFETCH)
 mx=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32)
 am=tl.zeros((BQ,D),tl.float32);ac=tl.zeros((BQ,16),tl.float32)
 if EXACT_COUNT:
  slots=tl.arange(0,tr.next_power_of_2(EXACT_COUNT))
  ids=tl.load(IDS+(h*(N//64)+t//(64//BQ))*EXACT_COUNT+slots,slots<EXACT_COUNT,0)
  bit=tl.where(slots<EXACT_COUNT,1<<(ids%32),0).to(tl.uint32)
  words=tl.sum(tl.where(tl.arange(0,16)[:,None]==ids[None,:]//32,bit[None,:],0),1)
 for first in range(tr.cdiv(P*4,BR)):
  rep=first*BR+r;page=first*(BR//4)+pages
  if EXACT_COUNT:
   word=tl.sum(tl.where(tl.arange(0,16)==first//2,words,0),0)
  else:
   word=tl.load(MASK+(h*(N//64)+(t*BQ)//64)*16+rep//4//32)
  omitted=((word>>(rep//4%32))&1)==0
  key=tl.load(KEY+(h*hp*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
  z=tl.dot(q,key)*(1.4426950408889634/math.sqrt(D))
  z=tl.where((rep[None,:]<P*4)&omitted[None,:],z,-float('inf'))
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
  if not EXACT_COUNT:
   le=tl.load(ELSE+h*N+qi)
   exact=tl.load(EXACT+(qi[:,None]*H+h)*D+d[None,:])
  lt=tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf'))
  rho=1./(1.+tl.exp2(le-lt))
  tl.store(OUT+(qi[:,None]*H+h)*D+d[None,:],exact*(1.-rho[:,None])+tail*rho[:,None])
 else:
  tl.store(MU+(h*N+qi[:,None])*D+d[None,:],am/safe[:,None])
  tl.store(CODE+(h*N+qi[:,None])*8+rank[None,:],ac/safe[:,None],rank[None,:]<8)
  tl.store(LSE+h*N+qi,tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf')))


