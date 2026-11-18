import math
import triton as tr
import triton.language as tl
from local_experiments.review_v1.tuned_attention import _av,_attn_fwd
@tr.jit
def partial(Q,KEY,MEAN,DETAIL,MASK,MU,CODE,LSE,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=64,BR:tl.constexpr=64, BF16_AV:tl.constexpr=False,MERGE:tl.constexpr=False,EXACT=None,ELSE=None,BASIS=None,OUT=None,HEAD_PAGES:tl.constexpr=0,K=None,V=None,IDS=None,EXACT_COUNT:tl.constexpr=0,QK_SCALE:tl.constexpr=0.,ANCHOR_SCALE:tl.constexpr=False,COLUMN_MASS:tl.constexpr=False,SPLIT:tl.constexpr=2):
 hp:tl.constexpr=HEAD_PAGES if HEAD_PAGES else P
 t=tl.program_id(0);h=tl.program_id(1);qi=t*BQ+tl.arange(0,BQ);d=tl.arange(0,D);r=tl.arange(0,BR);pages=tl.arange(0,BR//4);rank=tl.arange(0,16)
 q=tl.load(Q+(qi[:,None]*H+h)*D+d[None,:])
 part=tl.program_id(2)
 mx=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32)
 am=tl.zeros((BQ,D),tl.float32);ac=tl.zeros((BQ,16),tl.float32)
 if EXACT_COUNT:
  slots=tl.arange(0,tr.next_power_of_2(EXACT_COUNT))
  ids=tl.load(IDS+(h*(N//64)+t)*EXACT_COUNT+slots,slots<EXACT_COUNT,0)
  bit=tl.where(slots<EXACT_COUNT,1<<(ids%32),0).to(tl.uint32)
  words=tl.sum(tl.where(tl.arange(0,16)[:,None]==ids[None,:]//32,bit[None,:],0),1)
 for first in range(part*tr.cdiv(tr.cdiv(P*4,BR),SPLIT),tl.minimum((part+1)*tr.cdiv(tr.cdiv(P*4,BR),SPLIT),tr.cdiv(P*4,BR))):
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
 row=(part*H+h)*N+qi
 tl.store(MU+row[:,None]*D+d[None,:],am)
 tl.store(CODE+row[:,None]*8+rank[None,:],ac,rank[None,:]<8)
 tl.store(LSE+2*row,mx);tl.store(LSE+2*row+1,den)

@tr.jit
def merge_parts(EX,LE,MU,AC,STATE,BASIS,OUT,S:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);q=t*16+tl.arange(0,16);d=tl.arange(0,128);rank=tl.arange(0,16)
 row=h*4608+q
 mx=tl.load(STATE+2*row)
 for i in tl.static_range(1,S):
  mx=tl.maximum(mx,tl.load(STATE+2*(row+i*12*4608)))
 am=tl.full((16,128),0.,tl.float32);ac=tl.full((16,16),0.,tl.float32);den=tl.full((16,),0.,tl.float32)
 for i in tl.static_range(S):
  r=row+i*12*4608
  w=tl.exp2(tl.load(STATE+2*r)-mx)
  am+=w[:,None]*tl.load(MU+r[:,None]*128+d[None,:])
  ac+=w[:,None]*tl.load(AC+r[:,None]*8+rank[None,:],rank[None,:]<8,0)
  den+=w*tl.load(STATE+2*r+1)
 safe=tl.maximum(den,1.e-30)
 basis=tl.load(BASIS+(h*128+d[None,:])*8+rank[:,None],rank[:,None]<8,0).to(tl.float32)
 tail=am/safe[:,None]+tl.dot(ac/safe[:,None],basis,input_precision='tf32x3')
 lt=tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf'))
 le=tl.load(LE+q*12+h);rho=1./(1.+tl.exp2(le-lt))
 e=tl.load(EX+(q[:,None]*12+h)*128+d[None,:])
 tl.store(OUT+(q[:,None]*12+h)*128+d[None,:],e*(1.-rho[:,None])+tail*rho[:,None])
