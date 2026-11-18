import triton as tr
import triton.language as tl
@tr.jit
def encode(KEY,MEAN,DETAIL,OUT,P:tl.constexpr,HP:tl.constexpr,NT:tl.constexpr):
 h=tl.program_id(0);t=tl.program_id(1);part=tl.program_id(2)
 i=part*256+tl.arange(0,256)
 base=(h*NT+t)*10624
 if part<32:
  u=i//1024;m=i//256%4;l=i//8%32;j=i//4%2;b=i%4
  token=16*m+8*j+l//4
  k=16*u+2*(l%4)+8*(b//2)+b%2
  rep=t*64+token
  x=tl.load(KEY+(h*HP*4+rep)*128+k,rep<P*4,0)
  tl.store(OUT+base+i,x)
 elif part<40:
  ix=i-8192;f=ix//256;l=ix//8%32;j=ix//4%2;b=ix%4
  pg=t*16+2*(l%4)+8*(b//2)+b%2
  col=16*f+8*j+l//4
  x=tl.load(MEAN+(h*HP+pg)*128+col,pg<P,0)
  tl.store(OUT+base+i,x)
 else:
  ix=i-10240;u=ix//128;l=ix//4%32;b=ix%4
  pg=t*16+2*(l%4)+8*(b//2)+b%2
  x=tl.load(DETAIL+((h*HP+pg)*3+u)*8+l//4,(pg<P)&(u<3),0)
  tl.store(OUT+base+i,x,u<3)

from local_experiments.review_v1.tuned_attention import _attn_fwd
@tr.jit
def combine(Q,K,V,IDS,E,LE,MASS,MU,CODE,LT,BASIS,OUT,P:tl.constexpr,C:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);q=t*64+tl.arange(0,64)
 d=tl.arange(0,128);r=tl.arange(0,16);row=q*12+h
 mass=tl.load(MASS+row);e=tl.load(E+row[:,None]*128+d[None,:]);le=tl.load(LE+row)
 if tl.min(((mass>=1.)&(mass<1.21e24)).to(tl.int32),0)==0:
  e,le=_attn_fwd(Q,K,V,128**-.5,C,IDS,OUT,OUT,12,4608,P*64,72,128,64,64,HIGH_PRECISION_AV=True,RETURN_VALUES=True)
 i=h*4608+q
 basis=tl.load(BASIS+(h*128+d[None,:])*8+r[:,None],r[:,None]<8,0).to(tl.float32)
 code=tl.load(CODE+i[:,None]*8+r[None,:],r[None,:]<8,0)
 tail=tl.load(MU+i[:,None]*128+d[None,:])+tl.dot(code,basis,input_precision='tf32x3')
 rho=1./(1.+tl.exp2(le-tl.load(LT+i)))
 tl.store(OUT+row[:,None]*128+d[None,:],e*(1.-rho[:,None])+tail*rho[:,None])
