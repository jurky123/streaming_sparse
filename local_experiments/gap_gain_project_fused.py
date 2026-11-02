"""Fuse 16-probe projection and residual contributions; FP32/TF32x3 dot."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _project_contrib(PROB,QP,MASS,COV,MASK,OUT,ERR,P:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
    ht=tl.program_id(0);h=ht//T
    p=tl.program_id(1)*B+tl.arange(0,B)
    m=tl.arange(0,32);d=tl.arange(0,16)
    prob=tl.load(PROB+(ht*16+m[None,:])*P+p[:,None],(p[:,None]<P)&(m[None,:]<16),0)
    qp=tl.load(QP+ht*16*8+m[:,None]*8+d[None,:],(m[:,None]<16)&(d[None,:]<8),0)
    numerator=tl.dot(prob,qp,input_precision='tf32x3')
    mass=tl.load(MASS+ht*P+p,p<P,0)
    cov=tl.load(COV+(h*P+p[:,None])*8+d[None,:],(p[:,None]<P)&(d[None,:]<8),0).to(tl.float32)
    selected=tl.load(MASK+ht*P+p,p<P,0)
    qbar=tl.div_rn(numerator,16*tl.maximum(mass[:,None],1.e-12))
    value=mass[:,None]*(qbar*cov)
    addr=(ht*P+p[:,None])*8+d[None,:]
    valid=(p[:,None]<P)&(d[None,:]<8)
    tl.store(OUT+addr,value,valid)
    tl.store(ERR+addr,value*(~selected[:,None]).to(tl.float32),valid)

def contribution(prob,qp,mass,cov,mask,block=32):
    h,t,p=mass.shape
    assert prob.shape==(h,t,16,p) and qp.shape==(h,t,16,8)
    assert all(x.is_contiguous() for x in [prob,qp,mass,cov,mask])
    out=torch.empty(h,t,p,8,device=prob.device,dtype=torch.float32)
    err=torch.empty_like(out)
    _project_contrib[(h*t,tr.cdiv(p,block))](prob,qp,mass,cov,mask,out,err,p,t,block,num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out,err.sum(2)
