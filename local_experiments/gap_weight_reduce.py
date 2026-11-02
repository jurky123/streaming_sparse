"""Fuse probe/representative weighting without materializing H*T*M*P*4."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _weight(PROB,WITHIN,MASS,OUT,P:tl.constexpr,M:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);m=tl.arange(0,M);r=tl.arange(0,4)
    ht=i//P;p=i%P
    prob=tl.load(PROB+(ht[:,None]*M+m[None,:])*P+p[:,None],i[:,None]<N,0)
    w=tl.load(WITHIN+((ht[:,None,None]*M+m[None,:,None])*P+p[:,None,None])*4+r[None,None,:],i[:,None,None]<N,0)
    mass=tl.maximum(tl.load(MASS+i,i<N,1.),1.e-12)
    result=tl.div_rn(tl.sum(prob[:,:,None]*w,1)/M,mass[:,None])
    tl.store(OUT+i[:,None]*4+r[None,:],result,i[:,None]<N)
def weight(prob,within,mass,block=16):
    h,t,m,p=prob.shape;out=torch.empty(h,t,p,4,device=prob.device)
    _weight[(tr.cdiv(h*t*p,block),)](prob,within,mass,out,p,m,h*t*p,block,num_warps=4,enable_fp_fusion=False)
    return out
