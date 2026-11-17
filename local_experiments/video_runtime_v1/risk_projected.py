"""Fuse Risk features and M16 reduction without writing per-representative features."""
import torch
PAGE_BLOCK=16
WARPS=8
SUM_SEQ=False
FEATURE_SEQ=True
SQRT_LIB=True
PER_PROBE=False
KERNEL_STATS={}
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

@tr.jit
def sum4(x,M:tl.constexpr,P:tl.constexpr,SEQ:tl.constexpr):
    if SEQ:
        even,odd=tl.split(tl.reshape(x,(M,P,2,2)))
        x0,x2=tl.split(even);x1,x3=tl.split(odd)
        return ((x0+x1)+x2)+x3
    else:return tl.sum(x,2)

@tr.jit
def kernel(U,QN,PROB,ST,TH,C,OUT,T:tl.constexpr,M:tl.constexpr,P:tl.constexpr,BP:tl.constexpr,SEQ:tl.constexpr=False,FSEQ:tl.constexpr=False,LIB:tl.constexpr=False,PER:tl.constexpr=False):
    b=tl.program_id(0);head=b//T;p=tl.program_id(1)*BP+tl.arange(0,BP);m=tl.arange(0,M);r=tl.arange(0,4)
    base=ST+((head*P+p[None,:,None])*4+r[None,None,:])*12
    u=tl.load(U+((b*M+m[:,None,None])*P+p[None,:,None])*4+r[None,None,:],p[None,:,None]<P,0)
    u=tl.maximum(u,0)
    norm=tl.load(QN+b*M+m)
    beta=norm[:,None,None]*tl.load(base+8,p[None,:,None]<P,0).to(tl.float32)
    prob=tl.load(PROB+((b*M+m[:,None,None])*P+p[None,:,None])*4+r[None,None,:],p[None,:,None]<P,0)
    alpha=sum4(prob,M,BP,SEQ)
    mu=tl.load(ST+((head*P+p)*4)*12+11,p<P,0).to(tl.float32)
    c=tl.load(C+head*P+p,p<P,0)
    x0=alpha*mu[None,:]+sum4(tl.abs(prob-alpha[:,:,None]*.25)*tl.load(base+9,p[None,:,None]<P,0).to(tl.float32),M,BP,SEQ)
    root=lib.sqrt(u) if LIB else tl.sqrt(u)
    x1=sum4(prob*root*tl.load(base+10,p[None,:,None]<P,0).to(tl.float32),M,BP,SEQ)
    x2=c[None,:]*sum4(prob*beta,M,BP,SEQ)
    x3=c[None,:]*sum4(prob*(u+beta*beta),M,BP,SEQ)
    # Match the balanced four-term reduction in the original feature sum.
    f0=x0*tl.load(TH+head*4);f1=x1*tl.load(TH+head*4+1)
    f2=x2*tl.load(TH+head*4+2);f3=x3*tl.load(TH+head*4+3)
    value=((f0+f1)+f2)+f3 if FSEQ else (f0+f1)+(f2+f3)
    if PER:tl.store(OUT+(b*M+m[:,None])*P+p[None,:],value,p[None,:]<P)
    else:tl.store(OUT+b*P+p,tl.sum(value,0)/M,p<P)

@torch.library.custom_op('streaming_sparse::risk_projected',mutates_args=())
def reduce(u:torch.Tensor,qn:torch.Tensor,prob:torch.Tensor,stats:torch.Tensor,theta:torch.Tensor,c:torch.Tensor)->torch.Tensor:
    h,t,m,p=prob.shape[:4]
    u,qn,prob,stats,theta,c=[x.contiguous() for x in (u,qn,prob,stats,theta,c)]
    out=torch.empty((h,t,m,p) if PER_PROBE else (h,t,p),device=prob.device,dtype=torch.float32)
    compiled=kernel[(h*t,tr.cdiv(p,PAGE_BLOCK))](u,qn,prob,stats,theta,c,out,t,m,p,PAGE_BLOCK,SUM_SEQ,FEATURE_SEQ,SQRT_LIB,PER_PROBE,num_warps=WARPS,enable_fp_fusion=False)
    KERNEL_STATS[p]=dict(registers=compiled.n_regs,spills=compiled.n_spills,page_block=PAGE_BLOCK,warps=WARPS)
    return out

@reduce.register_fake
def _(u,qn,prob,stats,theta,c):return prob.new_empty(prob.shape[:4] if PER_PROBE else (prob.shape[0],prob.shape[1],prob.shape[3]))

def risk(probes,z,summary,theta):
    h,t,m,d=probes.shape;q=probes.bfloat16().float()
    prob=z.flatten(-2).softmax(-1).reshape_as(z)
    q2=q.square().reshape(h,t,m,8,16).sum(-1)/d;qn=q.norm(dim=-1)/(d**.5)
    bp=summary.stats[:,:,1,11].float();c=bp+bp.amax(-1,keepdim=True)
    u=torch.einsum("htmg,hprg->htmpr",q2,summary.stats.float()[...,:8])
    out=reduce(u,qn,prob,summary.stats,theta,c)
    return out.mean(2) if PER_PROBE else out
