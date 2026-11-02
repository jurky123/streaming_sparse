"""Page-local weighted V codes and fixed V variance; immutable historical writes."""
import torch
import triton as tr
import triton.language as tl
@tr.jit
def _pool(K,V,B,C,UK,UV,OUT,VAR,PK,H:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,START:tl.constexpr,OFFSET:tl.constexpr,RING:tl.constexpr,WEIGHTED:tl.constexpr,WRITE_PK:tl.constexpr):
    page=tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,D);r=tl.arange(0,8)
    slot=(START+page)%CAP if RING else page
    token=slot*64 if RING else OFFSET+page*64
    off=(token+t[:,None])*H*D+h*D+d[None,:]
    v=tl.load(V+off).to(tl.float32);mean=tl.sum(v,0)/64
    var=tl.maximum(tl.sum(tl.sum(v*v,0)/64-mean*mean,0)/D,0.)
    tl.store(VAR+h*CAP+slot,var)
    if WEIGHTED or WRITE_PK:
        k=tl.load(K+off).to(tl.float32)
        if WRITE_PK:
            kp=tl.sum(tl.reshape(k,(4,16,D)),1)/16
            tl.store(PK+((h*72+page)*4+tl.arange(0,4)[:,None])*D+d[None,:],kp)
    if WEIGHTED:
        uk=tl.load(UK+h*D+d);uv=tl.load(UV+h*D+d)
        score=tl.sum(k*uk[None,:],1)+tl.sum(v*uv[None,:],1)
        score=score-tl.sum(score,0)/64
        z=2*(2*tl.sigmoid(score)-1)
        a=tl.exp(z-tl.max(z,0));a=a/tl.sum(a,0)
        value=tl.sum(a[:,None]*v,0)-tl.load(C+h*D+d)
        basis=tl.load(B+h*D*8+d[:,None]*8+r[None,:]).to(tl.float32)
        code=tl.sum(value[:,None]*basis,0)
        tl.store(OUT+(h*CAP+slot)*8+r,code)

def pool(cache,basis,center,uk,uv,out,var,start=0,offset=0,ring=False,weighted=True,pk=None):
    basis=basis.contiguous();center=center.contiguous();uk=uk.contiguous();uv=uv.contiguous()
    cap=var.shape[1]
    _pool[(72,12)](cache['k'],cache['v'],basis,center,uk,uv,out,var,out if pk is None else pk,12,128,cap,start,offset,ring,weighted,pk is not None,num_warps=4)
