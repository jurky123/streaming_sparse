import triton
import triton.language as tl
@triton.jit
def pack_one(X,OUT,START,VALUE:tl.constexpr):
 p=tl.program_id(0)+START;h=tl.program_id(1)
 i=tl.program_id(2)*512+tl.arange(0,512);l=i//8%32
 r=2*(l%4)+8*(i//2%2)+i%2;c=8*(i//4%2)+l//4
 if VALUE:
  row=16*(i//2048)+r;d=16*(i//256%8)+c
 else:
  row=16*(i//256%4)+c;d=16*(i//1024)+r
 tl.store(OUT+(p*12+h)*8192+i,tl.load(X+((p*64+row)*12+h)*128+d))
