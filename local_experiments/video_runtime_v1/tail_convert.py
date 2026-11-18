"""Compute BF16 high/low together before dot-operand layout conversion."""
import math
import triton as tr
import triton.language as tl

@tr.jit
def _av_inline(a,b):
 hi,lo=tl.inline_asm_elementwise(
  "{ .reg .b16 h,l; .reg .f32 hf,r; cvt.rn.bf16.f32 h,$2; cvt.f32.bf16 hf,h; sub.rn.f32 r,$2,hf; cvt.rn.bf16.f32 l,r; mov.b16 $0,h; mov.b16 $1,l; }",
  constraints="=h,=h,f",args=[a],dtype=(tl.bfloat16,tl.bfloat16),is_pure=True,pack=1)
 return tl.dot(hi,b.to(tl.bfloat16))+tl.dot(lo,b.to(tl.bfloat16))

@tr.jit
def _tail_convert(Q,KEY,MEAN,DETAIL,MASK,MU,CODE,LSE,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=64,BR:tl.constexpr=64, BF16_AV:tl.constexpr=False):
 t=tl.program_id(0);h=tl.program_id(1);qi=t*BQ+tl.arange(0,BQ);d=tl.arange(0,D);r=tl.arange(0,BR);pages=tl.arange(0,BR//4);rank=tl.arange(0,16)
 q=tl.load(Q+(qi[:,None]*H+h)*D+d[None,:])
 mx=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32)
 am=tl.zeros((BQ,D),tl.float32);ac=tl.zeros((BQ,16),tl.float32)
 for first in range(tr.cdiv(P*4,BR)):
  rep=first*BR+r;page=first*(BR//4)+pages
  word=tl.load(MASK+(h*(N//64)+(t*BQ)//64)*16+rep//4//32)
  omitted=((word>>(rep//4%32))&1)==0
  key=tl.load(KEY+(h*P*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
  z=tl.dot(q,key)*(1.4426950408889634/math.sqrt(D))
  z=tl.where((rep[None,:]<P*4)&omitted[None,:],z,-float('inf'))
  nm=tl.maximum(mx,tl.max(z,1));fade=tl.exp2(mx-nm);w=tl.exp2(z-nm[:,None]);ww=tl.reshape(w,(BQ,BR//4,4));rr=tl.arange(0,4)
  alpha=tl.sum(ww,2)
  c0=tl.sum(ww*tl.where(rr<2,1.,-1.)[None,None,:],2)
  c1=tl.sum(ww*tl.where(rr%2==0,1.,-1.)[None,None,:],2)
  c2=tl.sum(ww*tl.where((rr==0)|(rr==3),1.,-1.)[None,None,:],2)
  mean=tl.load(MEAN+(h*P+page[:,None])*D+d[None,:],page[:,None]<P,0).to(tl.float32)
  x=tl.load(DETAIL+((h*P+page[:,None])*3)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
  y=tl.load(DETAIL+((h*P+page[:,None])*3+1)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
  v=tl.load(DETAIL+((h*P+page[:,None])*3+2)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
  if BF16_AV:
   am=am*fade[:,None]+_av_inline(alpha,mean)
   ac=ac*fade[:,None]+_av_inline(c0,x)+_av_inline(c1,y)+_av_inline(c2,v)
  else:
   am=am*fade[:,None]+tl.dot(alpha,mean,input_precision='tf32x3')
   ac=ac*fade[:,None]+tl.dot(c0,x,input_precision='tf32x3')+tl.dot(c1,y,input_precision='tf32x3')+tl.dot(c2,v,input_precision='tf32x3')
  den=den*fade+tl.sum(w,1);mx=nm
 safe=tl.maximum(den,1.e-30)
 tl.store(MU+(h*N+qi[:,None])*D+d[None,:],am/safe[:,None])
 tl.store(CODE+(h*N+qi[:,None])*8+rank[None,:],ac/safe[:,None],rank[None,:]<8)
 tl.store(LSE+h*N+qi,tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf')))

