"""Fuse FP64 log denominators and action scoring; materialize only fallback entries."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .scores_blocked import fallback_block
BLOCK=32
WARPS=4
SHARED_FIRST=False
KERNEL_STATS={}
@tr.jit
def logadd(x,y):
    nonfinite=((x!=x)|(tl.abs(x)==float('inf')))&(x==y)
    order=x>=y
    hi=tl.where(order,x,y);lo=tl.where(order,y,x)
    return tl.where(nonfinite,x,hi+lib.log1p(lib.exp(lo-hi)))

@tr.jit
def first_log(LB,RX,MID,N:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    lb=tl.load(LB+i//R,i<N,0);rx=tl.load(RX+i,i<N,0)
    tl.store(MID+i,logadd(lb,rx),i<N)

@tr.jit
def fused_scalar(LB,RX,AX,CAND,MID,DEN,STATS,DOT,MAX,OUT,FLAGS,R:tl.constexpr,A:tl.constexpr,M:tl.constexpr,BLOCK:tl.constexpr,SHARED:tl.constexpr=False):
    b=tl.program_id(1);action=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ri=action//A;ai=action%A;m=tl.arange(0,M)
    ix=(b*M+m[:,None])*(R*A)+action[None,:];valid=action<R*A
    lb=tl.load(LB+b*M+m)[:,None]
    rx=tl.load(RX+(b*M+m[:,None])*R+ri[None,:],valid[None,:],0)
    ax=tl.load(AX+(b*M+m[:,None])*A+ai[None,:],valid[None,:],0)
    lr=tl.load(CAND+((b*M+m[:,None])*(R+A)+ri[None,:])*34,valid[None,:],0)
    ap=CAND+((b*M+m[:,None])*(R+A)+R+ai[None,:])*34
    ea_log=tl.load(ap,valid[None,:],0)+tl.load(ap+1,valid[None,:],0)
    if SHARED:first=tl.load(MID+(b*M+m[:,None])*R+ri[None,:],valid[None,:],0)
    else:first=logadd(lb,rx)
    den=logadd(logadd(logadd(first,ax),lr),ea_log)
    maximum=tl.load(MAX+b*M+m)[:,None]
    ir=((b*M+m[:,None])*(R+A)+ri[None,:])*2;ia=((b*M+m[:,None])*(R+A)+R+ai[None,:])*2
    er=tl.load(STATS+ir,valid[None,:],0);nr=tl.load(STATS+ir+1,valid[None,:],0)
    ea=tl.load(STATS+ia,valid[None,:],0);na=tl.load(STATS+ia+1,valid[None,:],0)
    dot=tl.load(DOT+ix,valid[None,:],0);norm=nr+na-2*dot;linear=ea-er
    scale=lib.exp(maximum-den);first=2*scale*linear;second=scale*scale*norm;g=first-second
    bad=(maximum-den>80)|(tl.abs(g)>=float('inf'))|(g!=g)|(norm<0)
    bad=bad|((nr+na>0)&(tl.abs(norm)<1e-8*(nr+na)))
    bad=bad|((tl.abs(er)+tl.abs(ea)>0)&(tl.abs(linear)<1e-8*(tl.abs(er)+tl.abs(ea))))
    mag=tl.abs(first)+tl.abs(second);bad=bad|((mag>0)&(tl.abs(g)<1e-8*mag))
    total=tl.sum(g,0);absolute=tl.sum(tl.abs(g),0)
    fallback=(tl.sum(bad.to(tl.int32),0)>0)|((absolute>0)&(tl.abs(total)<1e-8*absolute))
    tl.store(DEN+ix,den,valid[None,:]&fallback[None,:])
    tl.store(OUT+b*R*A+action,total/M,valid);tl.store(FLAGS+b*R*A+action,fallback,valid)

@torch.library.custom_op('streaming_sparse::den_score_reduce',mutates_args=())
def den_score_reduce(state:torch.Tensor,cand:torch.Tensor,lb:torch.Tensor,rx:torch.Tensor,ro:torch.Tensor,ax:torch.Tensor,ao:torch.Tensor,maximum:torch.Tensor,stats:torch.Tensor,dot:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    b,m,r,a=dot.shape;d=state.shape[-1]
    state,cand,lb,rx,ro,ax,ao,maximum,stats,dot=[x.contiguous() for x in (state,cand,lb,rx,ro,ax,ao,maximum,stats,dot)]
    den=torch.empty_like(dot);out=torch.empty((b,r,a),device=dot.device,dtype=torch.float64);flags=torch.empty_like(out,dtype=torch.bool)
    mid=torch.empty_like(rx) if SHARED_FIRST else rx
    if SHARED_FIRST:first_log[(tr.cdiv(b*m*r,256),)](lb,rx,mid,b*m*r,r,256,num_warps=4,enable_fp_fusion=False)
    kernel=fused_scalar[(tr.cdiv(r*a,BLOCK),b)](lb,rx,ax,cand,mid,den,stats,dot,maximum,out,flags,r,a,m,BLOCK,SHARED_FIRST,num_warps=WARPS,enable_fp_fusion=False)
    fallback_block[(tr.cdiv(b*r*a,16),)](den,state,cand,lb,rx,ro,ax,ao,maximum,out,flags,b*r*a,r,a,m,d,16,ro.numel()==0,num_warps=4,enable_fp_fusion=False)
    KERNEL_STATS[(b,m,r,a)]=dict(registers=kernel.n_regs,spills=kernel.n_spills,block=BLOCK,warps=WARPS,shared_first=SHARED_FIRST)
    return out,flags
@den_score_reduce.register_fake
def _(state,cand,lb,rx,ro,ax,ao,maximum,stats,dot):
    shape=(dot.shape[0],dot.shape[2],dot.shape[3])
    return dot.new_empty(shape),dot.new_empty(shape,dtype=torch.bool)

@torch.library.custom_op('streaming_sparse::first_den',mutates_args=())
def first_den(lb:torch.Tensor,rx:torch.Tensor)->torch.Tensor:
    lb=lb.contiguous();rx=rx.contiguous();out=torch.empty_like(rx)
    first_log[(tr.cdiv(rx.numel(),256),)](lb,rx,out,rx.numel(),rx.shape[-1],256,num_warps=4,enable_fp_fusion=False)
    return out
@first_den.register_fake
def _(lb,rx):return torch.empty_like(rx)
