"""Contiguous page/representative vectorization, preserving four-accumulator order."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _weight(PROB,WITHIN,MASS,OUT,P:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);r=i%4;p=i//4%P;ht=i//(4*P)
    a=tl.full((B,),0.,tl.float32);b=a;c=a;d=a
    for j in tl.static_range(4):
        x=(ht*16+4*j)*P+p
        a=a+tl.load(PROB+x,i<N,0)*tl.load(WITHIN+x*4+r,i<N,0)
        b=b+tl.load(PROB+x+P,i<N,0)*tl.load(WITHIN+(x+P)*4+r,i<N,0)
        c=c+tl.load(PROB+x+2*P,i<N,0)*tl.load(WITHIN+(x+2*P)*4+r,i<N,0)
        d=d+tl.load(PROB+x+3*P,i<N,0)*tl.load(WITHIN+(x+3*P)*4+r,i<N,0)
    mass=tl.maximum(tl.load(MASS+i//4,i<N,1.),1.e-12)
    value=tl.div_rn((((a+b)+c)+d)/16.,mass)
    tl.store(OUT+i,value,i<N)
def weight(prob,within,mass,block=256,warps=4):
    h,t,m,p=prob.shape;assert m==16
    out=torch.empty(h,t,p,4,device=prob.device)
    _weight[(tr.cdiv(out.numel(),block),)](prob,within,mass,out,p,out.numel(),block,num_warps=warps,enable_fp_fusion=False)
    return out
