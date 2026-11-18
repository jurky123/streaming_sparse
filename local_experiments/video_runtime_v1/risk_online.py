"""Experimental online probabilities: materialize only row max and denominator."""
import torch
from .risk_bound import bound
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
def _risk_tile(b,U,QN,Z,NORM,ST,TH,C,OUT,T:tl.constexpr,M:tl.constexpr,P:tl.constexpr,BP:tl.constexpr,SEQ:tl.constexpr=False,FSEQ:tl.constexpr=False,LIB:tl.constexpr=False,PER:tl.constexpr=False,INLINE_U:tl.constexpr=False,LOGIT_TILE:tl.constexpr=0):
    head=b//T;p=tl.program_id(1)*BP+tl.arange(0,BP);m=tl.arange(0,M);r=tl.arange(0,4)
    base=ST+((head*P+p[None,:,None])*4+r[None,None,:])*12
    if INLINE_U:
        u=tl.full((M,BP,4),0,tl.float32)
        for g in tl.static_range(8):
            q2=tl.load(U+(b*M+m)*8+g)
            variance=tl.load(base+g,p[None,:,None]<P,0).to(tl.float32)
            u=tl.fma(q2[:,None,None],variance,u)
    else:
        u=tl.load(U+((b*M+m[:,None,None])*P+p[None,:,None])*4+r[None,None,:],p[None,:,None]<P,0)
    u=tl.maximum(u,0)
    norm=tl.load(QN+b*M+m)
    beta=norm[:,None,None]*tl.load(base+8,p[None,:,None]<P,0).to(tl.float32)
    if LOGIT_TILE:
        col=p[None,:,None]*4+r[None,None,:]
        offset=((b*tr.cdiv(P*4,LOGIT_TILE)+col//LOGIT_TILE)*M+m[:,None,None])*LOGIT_TILE+col%LOGIT_TILE
    else:
        offset=((b*M+m[:,None,None])*P+p[None,:,None])*4+r[None,None,:]
    logits=tl.load(Z+offset,p[None,:,None]<P,-float('inf')).to(tl.float32)
    mx=tl.load(NORM+(b*M+m)*2);den=tl.load(NORM+(b*M+m)*2+1)
    prob=tl.exp(logits-mx[:,None,None])/den[:,None,None]
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

@tr.jit
def kernel(U,QN,Z,NORM,ST,TH,C,OUT,T:tl.constexpr,M:tl.constexpr,P:tl.constexpr,BP:tl.constexpr,SEQ:tl.constexpr=False,FSEQ:tl.constexpr=False,LIB:tl.constexpr=False,PER:tl.constexpr=False,INLINE_U:tl.constexpr=False,LOGIT_TILE:tl.constexpr=0):
    _risk_tile(tl.program_id(0),U,QN,Z,NORM,ST,TH,C,OUT,T,M,P,BP,SEQ,FSEQ,LIB,PER,INLINE_U,LOGIT_TILE)

@torch.library.custom_op('streaming_sparse::risk_online',mutates_args=())
def reduce(u:torch.Tensor,qn:torch.Tensor,prob:torch.Tensor,norm:torch.Tensor,stats:torch.Tensor,theta:torch.Tensor,c:torch.Tensor)->torch.Tensor:
    h,t,m,p=prob.shape[:4]
    u,qn,prob,stats,theta,c=[x.contiguous() for x in (u,qn,prob,stats,theta,c)]
    out=torch.empty((h,t,m,p) if PER_PROBE else (h,t,p),device=prob.device,dtype=torch.float32)
    compiled=kernel[(h*t,tr.cdiv(p,PAGE_BLOCK))](u,qn,prob,norm,stats,theta,c,out,t,m,p,PAGE_BLOCK,SUM_SEQ,FEATURE_SEQ,SQRT_LIB,PER_PROBE,u.ndim==4,num_warps=WARPS,enable_fp_fusion=False)
    KERNEL_STATS[p]=dict(registers=compiled.n_regs,spills=compiled.n_spills,page_block=PAGE_BLOCK,warps=WARPS)
    return out

@reduce.register_fake
def _(u,qn,prob,norm,stats,theta,c):return prob.new_empty(prob.shape[:4] if PER_PROBE else (prob.shape[0],prob.shape[1],prob.shape[3]))

def risk(probes,z,summary,theta,inline=False,direct_bound=False):
    h,t,m,d=probes.shape;q=probes.bfloat16().float()
    norm=matched_normalizer(z) if inline else normalizer(z)
    q2=q.square().reshape(h,t,m,8,16).sum(-1)/d;qn=q.norm(dim=-1)/(d**.5)
    if direct_bound:c=bound(summary.stats)
    else:
        bp=summary.stats[:,:,1,11].float();c=bp+bp.amax(-1,keepdim=True)
    u=q2 if inline else torch.einsum("htmg,hprg->htmpr",q2,summary.stats.float()[...,:8])
    out=reduce(u,qn,z,norm,summary.stats,theta,c)
    return out.mean(2) if PER_PROBE else out

@tr.jit
def _normalize(Z,OUT,P:tl.constexpr,B:tl.constexpr,LOGIT_TILE:tl.constexpr=0):
 row=tl.program_id(0);i=tl.arange(0,B)
 if LOGIT_TILE:
  offset=((row//16*tr.cdiv(P*4,LOGIT_TILE)+i//LOGIT_TILE)*16+row%16)*LOGIT_TILE+i%LOGIT_TILE
 else:offset=row*P*4+i
 z=tl.load(Z+offset,i<P*4,-float('inf')).to(tl.float32)
 mx=tl.max(z,0);den=tl.sum(tl.exp(z-mx),0)
 tl.store(OUT+row*2,mx);tl.store(OUT+row*2+1,den)
@torch.library.custom_op('streaming_sparse::risk_row_norm',mutates_args=())
def normalizer(z:torch.Tensor)->torch.Tensor:
 out=z.new_empty((*z.shape[:3],2));p=z.shape[-2]
 _normalize[(z.shape[0]*z.shape[1]*z.shape[2],)](z,out,p,tr.next_power_of_2(p*4),num_warps=8,enable_fp_fusion=False)
 return out
@normalizer.register_fake
def _(z):return z.new_empty((*z.shape[:3],2))

@torch.library.custom_op('streaming_sparse::risk_matched_norm',mutates_args=())
def matched_normalizer(z:torch.Tensor)->torch.Tensor:
 out=z.new_empty((*z.shape[:3],2))
 p=z.shape[-2]
 _normalize[(out.numel()//2,)](z,out,p,tr.next_power_of_2(p*4),num_warps=8 if p<=256 else 16,enable_fp_fusion=False)
 return out

@matched_normalizer.register_fake
def _(z):return z.new_empty((*z.shape[:3],2))

def matched_risk(probes,z,summary,theta):
 return risk(probes,z,summary,theta,inline=True)


def bounded_risk(probes,z,summary,theta):
    return risk(probes,z,summary,theta,inline=True,direct_bound=True)
