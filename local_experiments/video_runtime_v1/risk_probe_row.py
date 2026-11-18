"""Stream a full M16 probe row through normalization, variance and all four terms."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

@tr.jit
def _row(Q2, QN, Z, ST, TH, C, OUT, T:tl.constexpr, P:tl.constexpr, B:tl.constexpr):
    row=tl.program_id(0)
    h=row//(T*16)
    i=tl.arange(0,B*4)
    p=tl.arange(0,B)
    z=tl.load(Z+row*P*4+i,i<P*4,-float('inf'))
    mass=tl.exp(z-tl.max(z,0))
    prob=mass/tl.sum(mass,0)
    alpha=tl.sum(tl.reshape(prob,(B,4)),1)
    norm=tl.load(QN+row)
    base=ST+(h*P*4+i)*12
    u=tl.full((B*4,),0,tl.float32)
    for g in tl.static_range(8):
        u=tl.fma(tl.load(Q2+row*8+g),tl.load(base+g,i<P*4,0).to(tl.float32),u)
    u=tl.maximum(u,0)
    beta=norm*tl.load(base+8,i<P*4,0).to(tl.float32)
    a0=tl.abs(tl.reshape(prob,(B,4))-alpha[:,None]*.25)
    x0=alpha*tl.load(ST+((h*P+p)*4)*12+11,p<P,0).to(tl.float32)+tl.sum(a0*tl.reshape(tl.load(base+9,i<P*4,0).to(tl.float32),(B,4)),1)
    x1=tl.sum(tl.reshape(prob*libdevice.sqrt(u)*tl.load(base+10,i<P*4,0).to(tl.float32),(B,4)),1)
    c=tl.load(C+h*P+p,p<P,0)
    x2=c*tl.sum(tl.reshape(prob*beta,(B,4)),1)
    x3=c*tl.sum(tl.reshape(prob*(u+beta*beta),(B,4)),1)
    v=((x0*tl.load(TH+h*4)+x1*tl.load(TH+h*4+1))+x2*tl.load(TH+h*4+2))+x3*tl.load(TH+h*4+3)
    tl.store(OUT+row*P+p,v,p<P)

@torch.library.custom_op('streaming_sparse::risk_probe_row',mutates_args=())
def rows(q2:torch.Tensor,qn:torch.Tensor,z:torch.Tensor,stats:torch.Tensor,theta:torch.Tensor,c:torch.Tensor)->torch.Tensor:
    h,t,m,p,_=z.shape
    out=z.new_empty((h,t,m,p))
    _row[(h*t*m,)](q2.contiguous(),qn.contiguous(),z.contiguous(),stats.contiguous(),theta.contiguous(),c.contiguous(),out,t,p,tr.next_power_of_2(p),num_warps=8,enable_fp_fusion=False)
    return out

@rows.register_fake
def _(q2,qn,z,stats,theta,c):return z.new_empty(z.shape[:-1])

def risk(probes,z,s,theta):
    q=probes.bfloat16().float()
    h,t,m,d=q.shape
    q2=q.square().reshape(h,t,m,8,16).sum(-1)/d
    qn=q.norm(dim=-1)/d**.5
    bp=s.stats[:,:,1,11].float()
    return rows(q2,qn,z,s.stats,theta,bp+bp.amax(-1,keepdim=True)).mean(2)
