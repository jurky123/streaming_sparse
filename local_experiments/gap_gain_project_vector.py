"""Direct FP32 16-probe contraction fused with residual contribution."""
import torch
import triton as tr
import triton.language as tl
@tr.jit
def _vector(PROB,QP,MASS,COV,MASK,OUT,ERR,P:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
    ht=tl.program_id(0);h=ht//T
    p=tl.program_id(1)*B+tl.arange(0,B);d=tl.arange(0,8)
    num=tl.full((B,8),0.,tl.float32)
    for m in tl.static_range(16):
        prob=tl.load(PROB+(ht*16+m)*P+p,p<P,0)
        qp=tl.load(QP+(ht*16+m)*8+d)
        num=tl.fma(prob[:,None],qp[None,:],num)
    mass=tl.load(MASS+ht*P+p,p<P,0)
    cov=tl.load(COV+(h*P+p[:,None])*8+d[None,:],p[:,None]<P,0).to(tl.float32)
    selected=tl.load(MASK+ht*P+p,p<P,0)
    qbar=tl.div_rn(num,16*tl.maximum(mass[:,None],1.e-12))
    value=mass[:,None]*(qbar*cov)
    addr=(ht*P+p[:,None])*8+d[None,:]
    tl.store(OUT+addr,value,p[:,None]<P)
    tl.store(ERR+addr,value*(~selected[:,None]).to(tl.float32),p[:,None]<P)
def contribution(prob,qp,mass,cov,mask,block=128):
    h,t,p=mass.shape
    assert prob.shape==(h,t,16,p) and qp.shape==(h,t,16,8)
    assert all(x.is_contiguous() for x in [prob,qp,mass,cov,mask])
    out=torch.empty(h,t,p,8,device=prob.device,dtype=torch.float32);err=torch.empty_like(out)
    _vector[(h*t,tr.cdiv(p,block))](prob,qp,mass,cov,mask,out,err,p,t,block,num_warps=4,enable_fp_fusion=False)
    return out,err.sum(2)
