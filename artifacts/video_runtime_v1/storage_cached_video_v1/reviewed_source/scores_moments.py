"""Endpoint inner products replace per-action vector work; unstable actions fall back.

All arithmetic and moments are FP64. Positive-group log denominators are supplied
unchanged. No page mass is subtracted to construct the new denominator.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .scores_packed import packed_kernel

BLOCKED=False
LAZY=False
FUSED_DEN=False
PRECOMPUTE_DEN=False

@tr.jit
def scalar_kernel(DEN,STATS,DOT,MAX,OUT,FLAGS,R:tl.constexpr,A:tl.constexpr,M:tl.constexpr):
    action=tl.program_id(0);b=action//(R*A);ri=(action//A)%R;ai=action%A;m=tl.arange(0,M)
    ix=((b*M+m)*R+ri)*A+ai
    den=tl.load(DEN+ix);maximum=tl.load(MAX+b*M+m)
    ir=((b*M+m)*(R+A)+ri)*2;ia=((b*M+m)*(R+A)+R+ai)*2
    er=tl.load(STATS+ir);nr=tl.load(STATS+ir+1);ea=tl.load(STATS+ia);na=tl.load(STATS+ia+1)
    dot=tl.load(DOT+ix);norm=nr+na-2*dot;linear=ea-er
    scale=lib.exp(maximum-den)
    first=2*scale*linear;second=scale*scale*norm;g=first-second
    bad=(maximum-den>80)|(tl.abs(g)>=float('inf'))|(g!=g)|(norm<0)
    bad=bad|((nr+na>0)&(tl.abs(norm)<1e-8*(nr+na)))
    bad=bad|((tl.abs(er)+tl.abs(ea)>0)&(tl.abs(linear)<1e-8*(tl.abs(er)+tl.abs(ea))))
    mag=tl.abs(first)+tl.abs(second)
    bad=bad|((mag>0)&(tl.abs(g)<1e-8*mag))
    total=tl.sum(g,0);absolute=tl.sum(tl.abs(g),0)
    fallback=(tl.sum(bad.to(tl.int32),0)>0)|((absolute>0)&(tl.abs(total)<1e-8*absolute))
    tl.store(OUT+action,total/M);tl.store(FLAGS+action,fallback)

@tr.jit
def fallback_kernel(DEN,STATE,CAND,LB,RX,RO,AX,AO,MAX,OUT,FLAGS,R:tl.constexpr,A:tl.constexpr,M:tl.constexpr,D:tl.constexpr,LAZY_MODE:tl.constexpr=False):
    if tl.load(FLAGS+tl.program_id(0)):
        packed_kernel(DEN,STATE,CAND,LB,RX,RO,AX,AO,MAX,OUT,R,A,M,D,LAZY=LAZY_MODE)

@torch.library.custom_op('streaming_sparse::moment_reduce',mutates_args=())
def moment_reduce(den:torch.Tensor,state:torch.Tensor,cand:torch.Tensor,lb:torch.Tensor,rx:torch.Tensor,ro:torch.Tensor,ax:torch.Tensor,ao:torch.Tensor,maximum:torch.Tensor,stats:torch.Tensor,dot:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    b,m,r,a=den.shape;d=state.shape[-1]
    out=torch.empty((b,r,a),device=den.device,dtype=torch.float64);flags=torch.empty_like(out,dtype=torch.bool)
    den,state,cand,lb,rx,ro,ax,ao,maximum,stats,dot=[x.contiguous() for x in (den,state,cand,lb,rx,ro,ax,ao,maximum,stats,dot)]
    if BLOCKED:
        from .scores_blocked import scalar_block,fallback_block
        scalar_block[(tr.cdiv(r*a,32),b)](den,stats,dot,maximum,out,flags,r,a,m,32,num_warps=4,enable_fp_fusion=False)
        fallback_block[(tr.cdiv(b*r*a,16),)](den,state,cand,lb,rx,ro,ax,ao,maximum,out,flags,b*r*a,r,a,m,d,16,ro.numel()==0,num_warps=4,enable_fp_fusion=False)
    else:
        scalar_kernel[(b*r*a,)](den,stats,dot,maximum,out,flags,r,a,m,num_warps=1,enable_fp_fusion=False)
        fallback_kernel[(b*r*a,)](den,state,cand,lb,rx,ro,ax,ao,maximum,out,flags,r,a,m,d,ro.numel()==0,num_warps=4,enable_fp_fusion=False)
    return out,flags

@moment_reduce.register_fake
def _(den,state,cand,lb,rx,ro,ax,ao,maximum,stats,dot):
    shape=(den.shape[0],den.shape[2],den.shape[3])
    return den.new_empty(shape),den.new_empty(shape,dtype=torch.bool)

def moments(den,state,cand,lb,rx,ro,ax,ao,maximum):
    r=den.shape[-2];H=state[:,:,0];E=state[:,:,1]
    lm,beta=cand[...,0].double(),cand[...,1].double();c,p=cand[...,2:18].double(),cand[...,18:].double()
    approx=(lm-maximum[...,None]).exp();exact=(lm+beta-maximum[...,None]).exp()
    diff=torch.where(beta.abs()<.5,approx*torch.expm1(beta.clamp(-.5,.5)),exact-approx)
    u=diff[...,None]*(c-H[:,:,None])+exact[...,None]*p
    stats=torch.stack(((u*E[:,:,None]).sum(-1),u.square().sum(-1)),-1)
    dot=u[:,:,:r]@u[:,:,r:].transpose(-1,-2)
    if FUSED_DEN:
        from .scores_den import den_score_reduce
        return den_score_reduce(state,cand,lb,rx,ro,ax,ao,maximum,stats,dot)
    return moment_reduce(den,state,cand,lb,rx,ro,ax,ao,maximum,stats,dot)
