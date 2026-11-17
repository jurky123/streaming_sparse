"""Latest full-mean + zero-sum detail attention; reuses repository page64 kernel.

BF16 Q/K/V and summaries, FP32 accumulation. The omitted AV supports
tf32x3 or two-term BF16 decomposition. Block sizes are explicit tuning options;
production candidate uses BQ=64, BR=64, BF16_AV=True. Numerical order differs.
"""
import math
import torch
import triton as tr
import triton.language as tl
from ..gap_exact_round_kernel import _attn_fwd
from ..high_sparse_runtime import _budget_mask

@tr.jit
def _av(a,b):
 hi=a.to(tl.bfloat16);lo=(a-hi.to(tl.float32)).to(tl.bfloat16)
 return tl.dot(hi,b.to(tl.bfloat16))+tl.dot(lo,b.to(tl.bfloat16))

@tr.jit
def _tail(Q,KEY,MEAN,DETAIL,MASK,MU,CODE,LSE,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=64,BR:tl.constexpr=64, BF16_AV:tl.constexpr=False):
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
   am=am*fade[:,None]+_av(alpha,mean)
   ac=ac*fade[:,None]+_av(c0,x)+_av(c1,y)+_av(c2,v)
  else:
   am=am*fade[:,None]+tl.dot(alpha,mean,input_precision='tf32x3')
   ac=ac*fade[:,None]+tl.dot(c0,x,input_precision='tf32x3')+tl.dot(c1,y,input_precision='tf32x3')+tl.dot(c2,v,input_precision='tf32x3')
  den=den*fade+tl.sum(w,1);mx=nm
 safe=tl.maximum(den,1.e-30)
 tl.store(MU+(h*N+qi[:,None])*D+d[None,:],am/safe[:,None])
 tl.store(CODE+(h*N+qi[:,None])*8+rank[None,:],ac/safe[:,None],rank[None,:]<8)
 tl.store(LSE+h*N+qi,tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf')))

@tr.jit
def _merge(EXACT,ELSE,MU,CODE,TLSE,BASIS,OUT,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);qi=t*16+tl.arange(0,16);d=tl.arange(0,D);r=tl.arange(0,16)
 code=tl.load(CODE+(h*N+qi[:,None])*8+r[None,:],r[None,:]<8,0)
 basis=tl.load(BASIS+(h*D+d[None,:])*8+r[:,None],r[:,None]<8,0).to(tl.float32)
 tail=tl.load(MU+(h*N+qi[:,None])*D+d[None,:])+tl.dot(code,basis,input_precision='tf32x3')
 le=tl.load(ELSE+h*N+qi);lt=tl.load(TLSE+h*N+qi);rho=1./(1.+tl.exp2(le-lt))
 exact=tl.load(EXACT+(qi[:,None]*H+h)*D+d[None,:])
 tl.store(OUT+(qi[:,None]*H+h)*D+d[None,:],exact*(1.-rho[:,None])+tail*rho[:,None])

@torch.no_grad()
def attention(q,k,v,s,basis,selection,*,bq=64,br=64,bf16_av=False,exact_stages=3):
 h,n,d=q.shape;p=k.shape[1]
 assert 'fixed_count' in selection,'Dynamic thresholds need a separate validated count-aware kernel'
 count=selection['fixed_count'];lut=selection['ids'][...,:count].contiguous()[None]
 qq=q.transpose(0,1).contiguous()[None];kk=k.flatten(1,2).transpose(0,1).contiguous()[None];vv=v.flatten(1,2).transpose(0,1).contiguous()[None]
 exact=torch.empty(qq.shape,device=q.device,dtype=torch.float32);le=torch.empty(h,n,device=q.device)
 _attn_fwd[(n//64,h)](qq,kk,vv,d**-.5,count,lut,le,exact,h,n,p*64,n//64,d,64,64,HIGH_PRECISION_AV=True,num_warps=4,num_stages=exact_stages)
 bits=torch.empty(h,n//64,16,device=q.device,dtype=torch.uint32)
 _budget_mask[(n//64,h)](lut,bits,n//64,count,tr.next_power_of_2(count),num_warps=4)
 mu=torch.empty(h,n,d,device=q.device);code=torch.empty(h,n,8,device=q.device);lt=torch.empty(h,n,device=q.device);out=torch.empty_like(qq)
 _tail[(n//bq,h)](qq,s.keys.contiguous(),s.mean.contiguous(),s.detail.contiguous(),bits,mu,code,lt,p,n,h,d,BQ=bq,BR=br,BF16_AV=bf16_av,num_warps=4,num_stages=2,enable_fp_fusion=False)
 _merge[(n//16,h)](exact,le,mu,code,lt,basis.bfloat16().contiguous(),out,n,h,d,num_warps=4,enable_fp_fusion=False)
 return out[0].transpose(0,1)
