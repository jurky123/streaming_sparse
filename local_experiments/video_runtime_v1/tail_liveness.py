"""Keep current arithmetic; consume mean and three detail terms sequentially."""
import math
import triton as tr
import triton.language as tl
from ..review_v1.tuned_attention import _av

@tr.jit
def _tail_late(Q,KEY,MEAN,DETAIL,MASK,MU,CODE,LSE,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=64,BR:tl.constexpr=64,BF16_AV:tl.constexpr=True):
 tl.static_assert(BF16_AV)
 t=tl.program_id(0);h=tl.program_id(1);qi=t*BQ+tl.arange(0,BQ);d=tl.arange(0,D);r=tl.arange(0,BR);pages=tl.arange(0,BR//4);rank=tl.arange(0,16)
 q=tl.load(Q+(qi[:,None]*H+h)*D+d[None,:]);mx=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32)
 am=tl.zeros((BQ,D),tl.float32);ac=tl.zeros((BQ,16),tl.float32)
 for first in range(tr.cdiv(P*4,BR)):
  rep=first*BR+r;page=first*(BR//4)+pages
  word=tl.load(MASK+(h*(N//64)+(t*BQ)//64)*16+rep//4//32);omitted=((word>>(rep//4%32))&1)==0
  key=tl.load(KEY+(h*P*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
  z=tl.dot(q,key)*(1.4426950408889634/math.sqrt(D));z=tl.where((rep[None,:]<P*4)&omitted[None,:],z,-float('inf'))
  nm=tl.maximum(mx,tl.max(z,1));fade=tl.exp2(mx-nm);w=tl.exp2(z-nm[:,None]);ww=tl.reshape(w,(BQ,BR//4,4));rr=tl.arange(0,4)
  alpha=tl.sum(ww,2);den=den*fade+tl.sum(w,1)
  mean=tl.load(MEAN+(h*P+page[:,None])*D+d[None,:],page[:,None]<P,0)
  am=am*fade[:,None]+_av(alpha,mean)
  c0=tl.sum(ww*tl.where(rr<2,1.,-1.)[None,None,:],2)
  x=tl.load(DETAIL+((h*P+page[:,None])*3)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0)
  ac=ac*fade[:,None]+_av(c0,x)
  c1=tl.sum(ww*tl.where(rr%2==0,1.,-1.)[None,None,:],2)
  y=tl.load(DETAIL+((h*P+page[:,None])*3+1)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0)
  ac=ac+_av(c1,y)
  c2=tl.sum(ww*tl.where((rr==0)|(rr==3),1.,-1.)[None,None,:],2)
  v=tl.load(DETAIL+((h*P+page[:,None])*3+2)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0)
  ac=ac+_av(c2,v);mx=nm
 safe=tl.maximum(den,1.e-30)
 tl.store(MU+(h*N+qi[:,None])*D+d[None,:],am/safe[:,None]);tl.store(CODE+(h*N+qi[:,None])*8+rank[None,:],ac/safe[:,None],rank[None,:]<8)
 tl.store(LSE+h*N+qi,tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf')))
