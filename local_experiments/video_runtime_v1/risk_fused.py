"""Fuse Risk features and M16 reduction without writing per-representative features."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def kernel(Q2,QN,PROB,ST,TH,C,OUT,T:tl.constexpr,M:tl.constexpr,P:tl.constexpr,BP:tl.constexpr):
    b=tl.program_id(0);head=b//T;p=tl.program_id(1)*BP+tl.arange(0,BP);m=tl.arange(0,M);r=tl.arange(0,4)
    base=ST+((head*P+p[None,:,None])*4+r[None,None,:])*12
    u=tl.full((M,BP,4),0,tl.float32)
    for g in tl.static_range(8):
        q=tl.load(Q2+(b*M+m)*8+g)
        st=tl.load(base+g,p[None,:,None]<P,0).to(tl.float32)
        u=u+q[:,None,None]*st
    u=tl.maximum(u,0)
    norm=tl.load(QN+b*M+m)
    beta=norm[:,None,None]*tl.load(base+8,p[None,:,None]<P,0).to(tl.float32)
    prob=tl.load(PROB+((b*M+m[:,None,None])*P+p[None,:,None])*4+r[None,None,:],p[None,:,None]<P,0)
    alpha=tl.sum(prob,2)
    mu=tl.load(ST+((head*P+p)*4)*12+11,p<P,0).to(tl.float32)
    c=tl.load(C+head*P+p,p<P,0)
    x0=alpha*mu[None,:]+tl.sum(tl.abs(prob-alpha[:,:,None]*.25)*tl.load(base+9,p[None,:,None]<P,0).to(tl.float32),2)
    x1=tl.sum(prob*tl.sqrt(u)*tl.load(base+10,p[None,:,None]<P,0).to(tl.float32),2)
    x2=c[None,:]*tl.sum(prob*beta,2)
    x3=c[None,:]*tl.sum(prob*(u+beta*beta),2)
    # Match the balanced four-term reduction in the original feature sum.
    f0=x0*tl.load(TH+head*4);f1=x1*tl.load(TH+head*4+1)
    f2=x2*tl.load(TH+head*4+2);f3=x3*tl.load(TH+head*4+3)
    value=(f0+f1)+(f2+f3)
    tl.store(OUT+b*P+p,tl.sum(value,0)/M,p<P)

@torch.library.custom_op('streaming_sparse::risk_fused',mutates_args=())
def reduce(q2:torch.Tensor,qn:torch.Tensor,prob:torch.Tensor,stats:torch.Tensor,theta:torch.Tensor,c:torch.Tensor)->torch.Tensor:
    h,t,m,p=prob.shape[:4]
    q2,qn,prob,stats,theta,c=[x.contiguous() for x in (q2,qn,prob,stats,theta,c)]
    out=torch.empty((h,t,p),device=prob.device,dtype=torch.float32)
    kernel[(h*t,tr.cdiv(p,16))](q2,qn,prob,stats,theta,c,out,t,m,p,16,num_warps=4,enable_fp_fusion=False)
    return out

@reduce.register_fake
def _(q2,qn,prob,stats,theta,c):return prob.new_empty((prob.shape[0],prob.shape[1],prob.shape[3]))

def risk(probes,z,summary,theta):
    h,t,m,d=probes.shape;q=probes.bfloat16().float()
    prob=z.flatten(-2).softmax(-1).reshape_as(z)
    q2=q.square().reshape(h,t,m,8,16).sum(-1)/d;qn=q.norm(dim=-1)/(d**.5)
    bp=summary.stats[:,:,1,11].float();c=bp+bp.amax(-1,keepdim=True)
    return reduce(q2,qn,prob,summary.stats,theta,c)
