"""Copy-only fusion: preserve the original prototype reduction kernel exactly."""
import triton
import triton.language as tl
@triton.jit
def _copy(K,V,O,CK,CV,W,CAP:tl.constexpr):
    p=tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,128)
    ids=tl.load(O+(h*72+p)*64+t);src=(ids[:,None]*12+h)*128+d[None,:]
    slot=(W+p)%CAP;dst=((slot*64+t[:,None])*12+h)*128+d[None,:]
    tl.store(CK+dst,tl.load(K+src));tl.store(CV+dst,tl.load(V+src))
def copy(k,v,order,cache,write,cap):_copy[(72,12)](k,v,order,cache['k'],cache['v'],write,cap,num_warps=4)
