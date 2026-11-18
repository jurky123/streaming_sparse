"""Factor each page's Risk contribution by the shared mass of its probe row."""
import math
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


CHANNEL_MAJOR_STATS = False


@triton.jit
def _partial(Q,K,U,QN,ST,TH,C,VAL,NORM,T:tl.constexpr,P:tl.constexpr,BP:tl.constexpr,CHANNEL_MAJOR:tl.constexpr=False):
    b=tl.program_id(0)
    block=tl.program_id(1)
    head=b//T
    m=tl.arange(0,16)
    n=block*(4*BP)+tl.arange(0,4*BP)
    p=block*BP+tl.arange(0,BP)
    d=tl.arange(0,32)
    rep=tl.arange(0,4)
    acc=tl.full((16,4*BP),0,tl.float32)
    for j in range(4):
        dim=j*32+d
        q=tl.load(Q+(b*16+m[:,None])*128+dim[None,:])
        k=tl.load(K+(head*P*4+n[None,:])*128+dim[:,None],n[None,:]<P*4,0)
        acc=tl.dot(q,k,acc)
    logits=tl.where(n[None,:]<P*4,acc*(128**-.5),-float('inf'))
    mx=tl.max(logits,1)
    mass=tl.exp(logits-mx[:,None])
    den=tl.sum(mass,1)
    w=tl.reshape(mass,(16,BP,4))
    if CHANNEL_MAJOR:
        st=ST+head*12*P*4+p[None,:,None]*4+rep[None,None,:]
        stride=P*4
        mu_ptr=ST+(head*12+11)*P*4+p*4
    else:
        st=ST+((head*P+p[None,:,None])*4+rep[None,None,:])*12
        stride=1
        mu_ptr=ST+((head*P+p)*4)*12+11
    u=tl.full((16,BP,4),0,tl.float32)
    for g in tl.static_range(8):
        q2=tl.load(U+(b*16+m)*8+g)
        var=tl.load(st+g*stride,p[None,:,None]<P,0).to(tl.float32)
        u=tl.fma(q2[:,None,None],var,u)
    u=tl.maximum(u,0.)
    beta=tl.load(QN+b*16+m)[:,None,None]*tl.load(st+8*stride,p[None,:,None]<P,0).to(tl.float32)
    alpha=tl.sum(w,2)
    mu=tl.load(mu_ptr,p<P,0).to(tl.float32)
    c=tl.load(C+head*P+p,p<P,0)
    x0=alpha*mu[None,:]+tl.sum(tl.abs(w-alpha[:,:,None]*.25)*tl.load(st+9*stride,p[None,:,None]<P,0).to(tl.float32),2)
    x1=tl.sum(w*libdevice.sqrt(u)*tl.load(st+10*stride,p[None,:,None]<P,0).to(tl.float32),2)
    x2=c[None,:]*tl.sum(w*beta,2)
    x3=c[None,:]*tl.sum(w*(u+beta*beta),2)
    val=((x0*tl.load(TH+head*4)+x1*tl.load(TH+head*4+1))+x2*tl.load(TH+head*4+2))+x3*tl.load(TH+head*4+3)
    row=(b*triton.cdiv(P,BP)+block)*16+m
    tl.store(VAL+row[:,None]*BP+tl.arange(0,BP)[None,:],val)
    tl.store(NORM+row*2,mx)
    tl.store(NORM+row*2+1,den)


@triton.jit
def _finish(VAL,NORM,OUT,P:tl.constexpr,BP:tl.constexpr,N:tl.constexpr):
    b=tl.program_id(0)
    block=tl.program_id(1)
    n=tl.arange(0,N)
    m=tl.arange(0,16)
    p=block*BP+tl.arange(0,BP)
    nb=triton.cdiv(P,BP)
    ptr=NORM+((b*nb+n[:,None])*16+m[None,:])*2
    mx=tl.load(ptr,n[:,None]<nb,-float('inf'))
    den=tl.load(ptr+1,n[:,None]<nb,0)
    maximum=tl.max(mx,0)
    total=tl.sum(den*tl.exp(mx-maximum[None,:]),0)
    row=(b*nb+block)*16+m
    scale=tl.exp(tl.load(NORM+row*2)-maximum)/total
    val=tl.load(VAL+row[:,None]*BP+tl.arange(0,BP)[None,:])
    tl.store(OUT+b*P+p,tl.sum(val*scale[:,None],0)/16.,p<P)

import torch


def _score(q,k,u,qn,st,theta,c):
    heads,tiles,_,_=q.shape
    pages=st.shape[1]
    width=128
    blocks=triton.cdiv(pages,width)
    val=q.new_empty((heads*tiles,blocks,16,width),dtype=torch.float32)
    stats=q.new_empty((heads*tiles,blocks,16,2),dtype=torch.float32)
    out=q.new_empty((heads,tiles,pages),dtype=torch.float32)
    partial_stats=st.flatten(1,2).transpose(1,2).contiguous() if CHANNEL_MAJOR_STATS else st.contiguous()
    _partial[(heads*tiles,blocks)](q.contiguous(),k.contiguous(),u.contiguous(),qn.contiguous(),partial_stats,
                                   theta.contiguous(),c.contiguous(),val,stats,tiles,pages,width,CHANNEL_MAJOR_STATS,
                                   num_warps=4,enable_fp_fusion=False)
    _finish[(heads*tiles,blocks)](val,stats,out,pages,width,triton.next_power_of_2(blocks),
                                  num_warps=4,enable_fp_fusion=False)
    return out


@torch.library.custom_op('streaming_sparse::risk_partial_mass', mutates_args=())
def reduce(q:torch.Tensor,k:torch.Tensor,u:torch.Tensor,qn:torch.Tensor,st:torch.Tensor,
           theta:torch.Tensor,c:torch.Tensor,own:torch.Tensor|None=None,cuda_flags:bool=False)->torch.Tensor:
    out=_score(q,k,u,qn,st,theta,c)
    if own is not None:
        from .risk_recheck import repair
        repair(q,k,u,qn,st,theta,c,own,out,cuda_flags=cuda_flags)
    return out


@reduce.register_fake
def _(q,k,u,qn,st,theta,c,own=None,cuda_flags=False):
    return q.new_empty((q.shape[0],q.shape[1],st.shape[1]),dtype=torch.float32)


def score(q,summary,theta,own=None,cuda_flags=False):
    from .risk_bound import bound
    probes=q.float()
    u=probes.square().reshape(*q.shape[:3],8,16).sum(-1)/128
    qn=probes.norm(dim=-1)/math.sqrt(128)
    return reduce(q,summary.keys,u,qn,summary.stats,theta,bound(summary.stats),own,cuda_flags)

@torch.library.custom_op('streaming_sparse::rechecked_route',mutates_args=())
def route(q:torch.Tensor,k:torch.Tensor,u:torch.Tensor,qn:torch.Tensor,st:torch.Tensor,
          theta:torch.Tensor,c:torch.Tensor,own:torch.Tensor,radix_select:bool=False)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    from .risk_recheck import repair
    q,k,u,qn,st,theta,c=[x.contiguous() for x in (q,k,u,qn,st,theta,c)]
    risk=_score(q,k,u,qn,st,theta,c)
    ids,mask=repair(q,k,u,qn,st,theta,c,own,risk,select_pages=True,radix_select=radix_select)
    return risk,ids,mask


@route.register_fake
def _(q,k,u,qn,st,theta,c,own,radix_select=False):
    h,t=q.shape[:2]
    p=st.shape[1]
    return (q.new_empty((h,t,p),dtype=torch.float32),q.new_empty((h,t,p//10),dtype=torch.int32),
            q.new_empty((h,t,p),dtype=torch.bool))


def select_scores(q,s,theta,own,radix_select=False):
    from .risk_bound import bound
    probes=q.float()
    u=probes.square().reshape(*q.shape[:3],8,16).sum(-1)/128
    qn=probes.norm(dim=-1)/math.sqrt(128)
    return route(q,s.keys,u,qn,s.stats,theta,bound(s.stats),own,radix_select)
