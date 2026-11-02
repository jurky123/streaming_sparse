"""Fuse probe/representative weighting without materializing H*T*M*P*4."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _weight(PROB,WITHIN,MASS,OUT,P:tl.constexpr,M:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);m=tl.arange(0,M);r=tl.arange(0,4)
    ht=i//P;p=i%P
    g=tl.arange(0,4)
    acc=tl.full((B,4,4),0.,tl.float32)
    for j in tl.static_range(4):
        probe=g+4*j
        prob=tl.load(PROB+(ht[:,None]*M+probe[None,:])*P+p[:,None],i[:,None]<N,0)
        w=tl.load(WITHIN+((ht[:,None,None]*M+probe[None,:,None])*P+p[:,None,None])*4+r[None,None,:],i[:,None,None]<N,0)
        acc=acc+prob[:,:,None]*w
    a=tl.sum(tl.where(g[None,:,None]==0,acc,0.),1)
    b=tl.sum(tl.where(g[None,:,None]==1,acc,0.),1)
    c=tl.sum(tl.where(g[None,:,None]==2,acc,0.),1)
    d=tl.sum(tl.where(g[None,:,None]==3,acc,0.),1)
    mass=tl.maximum(tl.load(MASS+i,i<N,1.),1.e-12)
    result=tl.div_rn((((a+b)+c)+d)/M,mass[:,None])
    tl.store(OUT+i[:,None]*4+r[None,:],result,i[:,None]<N)
def weight(prob,within,mass,block=16):
    h,t,m,p=prob.shape;assert m==16;out=torch.empty(h,t,p,4,device=prob.device)
    _weight[(tr.cdiv(h*t*p,block),)](prob,within,mass,out,p,m,h*t*p,block,num_warps=4,enable_fp_fusion=False)
    return out
