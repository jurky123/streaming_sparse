"""QK and softmax statistics for sixteen probes sharing the same representative keys."""
import triton
import triton.language as tl


@triton.jit
def full_rows(Q,K,Z,NORM,T:tl.constexpr,COLS:tl.constexpr,B:tl.constexpr):
    tile=tl.program_id(0)
    head=tl.program_id(1)
    m=tile*16+tl.arange(0,16)
    n=tl.arange(0,B)
    d=tl.arange(0,32)
    acc=tl.full((16,B),0,tl.float32)
    for j in range(4):
        dim=j*32+d
        q=tl.load(Q+((head*T*16+m[:,None])*128+dim[None,:]))
        k=tl.load(K+((head*COLS+n[None,:])*128+dim[:,None]),n[None,:]<COLS,0)
        acc+=tl.dot(q,k)
    z=acc*(128**-.5)
    row=head*T*16+m
    tl.store(Z+row[:,None]*COLS+n[None,:],z,n[None,:]<COLS)
    masked=tl.where(n[None,:]<COLS,z,-float('inf'))
    mx=tl.max(masked,1)
    den=tl.sum(tl.exp(masked-mx[:,None]),1)
    tl.store(NORM+row*2,mx)
    tl.store(NORM+row*2+1,den)
