"""Gather probe Q and project onto BF16-rounded V basis in one launch."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _gather_project(Q,OFF,BASIS,SELECTED,QP,T:tl.constexpr,M:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr):
    h=tl.program_id(1);i=tl.program_id(0)*BQ+tl.arange(0,BQ)
    d=tl.arange(0,D);r=tl.arange(0,16)
    offset=tl.load(OFF+i%M,i<T*M,0);token=(i//M)*64+offset
    q=tl.load(Q+token[:,None]*H*D+h*D+d[None,:],i[:,None]<T*M,0)
    basis=tl.load(BASIS+h*D*8+d[:,None]*8+r[None,:],r[None,:]<8,0).to(tl.bfloat16)
    value=tl.dot(q,basis)*0.08838834764831843
    tl.store(SELECTED+(h*T*M+i[:,None])*D+d[None,:],q.to(tl.float32),i[:,None]<T*M)
    tl.store(QP+(h*T*M+i[:,None])*8+r[None,:],value,(i[:,None]<T*M)&(r[None,:]<8))
def gather_project(q,offsets,basis):
    _,n,h,d=q.shape;t=n//64;m=offsets.numel()
    assert q.is_contiguous() and basis.is_contiguous() and q.dtype==torch.bfloat16 and d==128
    selected=torch.empty(h,t,m,d,device=q.device,dtype=torch.float32)
    qp=torch.empty(h,t,m,8,device=q.device,dtype=torch.float32)
    _gather_project[(tr.cdiv(t*m,32),h)](q,offsets,basis,selected,qp,t,m,h,d,32,num_warps=4,num_stages=2,enable_fp_fusion=False)
    return selected,qp
