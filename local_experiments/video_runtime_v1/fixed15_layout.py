import triton as tr
import triton.language as tl

@tr.jit
def pack_range(K,V,PK,PV,START):
    # Use a single specialization for both portions of a wrapped history commit.
    page=tl.program_id(0)+START
    h=tl.program_id(1)
    i=tl.program_id(2)*512+tl.arange(0,512)
    lane=i//8%32
    red=2*(lane%4)+8*(i//2%2)+i%2
    col=8*(i//4%2)+lane//4
    kc=16*(i//256%4)+col;kr=16*(i//1024)+red
    vr=16*(i//2048)+red;vc=16*(i//256%8)+col
    base=(page*64*12+h)*128
    dest=(page*12+h)*8192+i
    tl.store(PK+dest,tl.load(K+base+kc*1536+kr))
    tl.store(PV+dest,tl.load(V+base+vr*1536+vc))

@tr.jit
def encode(KEY,MEAN,DETAIL,OUT,P:tl.constexpr,HP:tl.constexpr,NT:tl.constexpr,DIRECT_MASS:tl.constexpr=False):
 # Historical experiments use the original ordering. Current CUDA prepare uses DIRECT_MASS=True.
 h=tl.program_id(0);t=tl.program_id(1);part=tl.program_id(2)
 i=part*256+tl.arange(0,256)
 base=(h*NT+t)*10624
 if part<32:
  u=i//1024;m=i//256%4;l=i//8%32;j=i//4%2;b=i%4
  token=16*m+8*j+l//4
  k=16*u+2*(l%4)+8*(b//2)+b%2
  if DIRECT_MASS:
   page=8*(token//32)+2*((token%8)//2)+token%32//16
   rep=t*64+4*page+2*((token//8)%2)+token%2
  else:
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

from local_experiments.review_v1.tuned_attention import _tail
@tr.jit
def retry(FLAGS,Q,K,V,IDS,KEY,MEAN,DETAIL,BASIS,OUT,P:tl.constexpr,HP:tl.constexpr,C:tl.constexpr):
 if tl.load(FLAGS+tl.program_id(1)*72+tl.program_id(0)):
  _tail(Q,KEY,MEAN,DETAIL,None,None,None,None,P,4608,12,128,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=BASIS,OUT=OUT,HEAD_PAGES=HP,K=K,V=V,IDS=IDS,ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=C,QK_SCALE=128**-.5)
