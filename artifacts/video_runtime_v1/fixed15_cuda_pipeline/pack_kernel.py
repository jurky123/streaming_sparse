import triton
import triton.language as tl

@triton.jit
def pack_pages(K,V,PK,PV,START:tl.constexpr):
    page=tl.program_id(0)+START
    h=tl.program_id(1)
    i=tl.program_id(2)*512+tl.arange(0,512)
    lane=i//8%32
    reduced=2*(lane%4)+8*(i//2%2)+i%2
    column=8*(i//4%2)+lane//4
    kc=16*(i//256%4)+column
    # AV has eight output fragment-pairs per reduction slice.
    vr=16*(i//2048)+reduced
    vc=16*(i//256%8)+column
    # QK has four output fragment-pairs per reduction slice.
    kr=16*(i//1024)+reduced
    base=(page*64*12+h)*128
    dest=(page*12+h)*8192+i
    tl.store(PK+dest,tl.load(K+base+kc*1536+kr))
    tl.store(PV+dest,tl.load(V+base+vr*1536+vc))

@triton.jit
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
