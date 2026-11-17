"""Read cached page coefficients and reduce responses without expanded P×16 outputs.

Only H/E/background and 48 candidate endpoints leave the kernel. All mixture
normalization and action inputs remain FP64; neural readout arithmetic is FP32.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .scores import exclude_group
from .scores_action import action_reduce

@tr.jit
def feature(U,A,row,head,p,P:tl.constexpr,J:tl.constexpr):
    f=tl.full(p.shape,0,tl.float32)
    for i in tl.static_range(8):
        u=tl.load(U+row*8+i)
        a=tl.load(A+head*P*32+p*32+i*4+J,p<P,0)
        f=f+u*a
    return lib.tanh(f)

@tr.jit
def reader_kernel(U,CO,A,MEAN,DETAIL,Z,LM,MASK,MAP,STATE,LOGSTATE,CAND,
                  T:tl.constexpr,M:tl.constexpr,P:tl.constexpr,K:tl.constexpr,S:tl.constexpr,NS:tl.constexpr,AGG32:tl.constexpr):
    dtype:tl.constexpr=tl.float32 if AGG32 else tl.float64
    row=tl.program_id(0);b=row//M;head=b//T
    shard=tl.program_id(1)
    p=shard*S+tl.arange(0,S);d=tl.arange(0,16)
    f0=feature(U,A,row,head,p,P,0);f1=feature(U,A,row,head,p,P,1)
    f2=feature(U,A,row,head,p,P,2);f3=feature(U,A,row,head,p,P,3)
    base=CO+head*P*85+p[:,None]*85+d[None,:]
    rp=f0[:,None]*tl.load(base+17,p[:,None]<P,0)
    rp=rp+f1[:,None]*tl.load(base+34,p[:,None]<P,0)
    rp=rp+f2[:,None]*tl.load(base+51,p[:,None]<P,0)
    rp=rp+f3[:,None]*tl.load(base+68,p[:,None]<P,0)
    rp=(rp+tl.load(base,p[:,None]<P,0)).to(dtype)
    cb=CO+head*P*85+p*85+16
    beta=f0*tl.load(cb+17,p<P,0)+f1*tl.load(cb+34,p<P,0)
    beta=beta+f2*tl.load(cb+51,p<P,0);beta=beta+f3*tl.load(cb+68,p<P,0)
    beta=(beta+tl.load(cb,p<P,0)).to(dtype)
    z0=tl.load(Z+(row*P+p)*4,p<P,0).to(dtype)
    z1=tl.load(Z+(row*P+p)*4+1,p<P,0).to(dtype)
    z2=tl.load(Z+(row*P+p)*4+2,p<P,0).to(dtype)
    z3=tl.load(Z+(row*P+p)*4+3,p<P,0).to(dtype)
    zm=tl.maximum(tl.maximum(z0,z1),tl.maximum(z2,z3))
    w0=lib.exp(z0-zm);w1=lib.exp(z1-zm);w2=lib.exp(z2-zm);w3=lib.exp(z3-zm)
    total=w0+w1+w2+w3;w0=w0/total;w1=w1/total;w2=w2/total;w3=w3/total
    c0=w0+w1-w2-w3;c1=w0-w1+w2-w3;c2=w0-w1-w2+w3
    db=DETAIL+head*P*48+p[:,None]*48+d[None,:]
    approx=c0[:,None]*tl.load(db,p[:,None]<P,0).to(dtype)
    approx=approx+c1[:,None]*tl.load(db+16,p[:,None]<P,0).to(dtype)
    approx=approx+c2[:,None]*tl.load(db+32,p[:,None]<P,0).to(dtype)
    approx=approx+tl.load(MEAN+head*P*16+p[:,None]*16+d[None,:],p[:,None]<P,0).to(dtype)
    lm=tl.load(LM+row*P+p,p<P,-float('inf')).to(dtype)
    # All mixtures and endpoints share this translation; it cancels in deltas.
    center=approx
    selected=tl.load(MASK+b*P+p,p<P,0);slot=tl.load(MAP+b*P+p,p<P,-1)
    le=lm+beta;logs=tl.where(selected,le,lm)
    mx=tl.max(logs,0);me=tl.max(le,0);bg=tl.where(slot>=0,-float('inf'),logs);mb=tl.max(bg,0)
    wc=lib.exp(logs-mx);wc=wc/tl.sum(wc,0)
    wt=lib.exp(le-me);wt=wt/tl.sum(wt,0)
    wb=lib.exp(bg-tl.where(mb>-float('inf'),mb,0.));zb=tl.sum(wb,0);wb=wb/tl.where(zb>0,zb,1.)
    current=center+tl.where(selected[:,None],rp,0.)
    h=tl.sum(wc[:,None]*current,0);target=tl.sum(wt[:,None]*(center+rp),0)
    ob=tl.sum(wb[:,None]*current,0)
    tl.store(STATE+(row*NS+shard)*48+d,h);tl.store(STATE+(row*NS+shard)*48+16+d,target);tl.store(STATE+(row*NS+shard)*48+32+d,ob)
    tl.store(LOGSTATE+(row*NS+shard)*4,mx+lib.log(tl.sum(lib.exp(logs-mx),0)))
    tl.store(LOGSTATE+(row*NS+shard)*4+1,me+lib.log(tl.sum(lib.exp(le-me),0)))
    tl.store(LOGSTATE+(row*NS+shard)*4+2,mb+lib.log(zb))
    tl.store(LOGSTATE+(row*NS+shard)*4+3,mx)
    cp=CAND+(row*K+slot)*34
    valid=(p<P)&(slot>=0)
    tl.store(cp,lm,valid);tl.store(cp+1,beta,valid)
    tl.store(cp[:,None]+2+d[None,:],center,valid[:,None])
    tl.store(cp[:,None]+18+d[None,:],rp,valid[:,None])

KERNEL_STATS={}
WARPS=8
SHARD=128
AGG32=False
GROUP=1
PACKED=False
FMA=False
MOMENTS=False
PAGE32=False
LAYOUT=False
SPLIT=False
VECTOR_WARPS=4
VECTOR_WIDTH=16
VECTOR_MAXNREG=0
VECTOR_PACK=False
LAYOUT_KIND="a"

@torch.library.custom_op('streaming_sparse::reader_reduce',mutates_args=())
def reader_reduce(u:torch.Tensor,co:torch.Tensor,A:torch.Tensor,mean:torch.Tensor,detail:torch.Tensor,z:torch.Tensor,lm:torch.Tensor,mask:torch.Tensor,mapping:torch.Tensor,candidates:int,shard:int,warps:int,aggregate_float:bool,query_group:int,fma:bool)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    h,t,m,p=z.shape[:4];b=h*t
    if aggregate_float and PAGE32:mean=mean.float();detail=detail.float()
    kernel=reader_kernel;launch={}
    if LAYOUT:
        pack_co=LAYOUT_KIND in ("co","both");pack_a=LAYOUT_KIND in ("a","both")
        if pack_co:co=torch.cat((co[...,:16].reshape(h,-1),co[...,16].transpose(1,2).reshape(h,-1)),1)
        if pack_a:A=A.permute(0,2,3,1).contiguous()
        launch=dict(PACK_CO=pack_co,PACK_A=pack_a)
        from .reader_layout import reader_kernel as kernel
    args=[x.contiguous() for x in (u,co,A,mean,detail,z,lm,mask,mapping)]
    ns=tr.cdiv(p,shard)
    state=torch.empty((b,m,ns,3,16),device=u.device,dtype=torch.float64)
    logs=torch.empty((b,m,ns,4),device=u.device,dtype=torch.float64)
    cand=torch.empty((b,m,candidates,34),device=u.device,dtype=torch.float64)
    if SPLIT:
        assert aggregate_float and LAYOUT and LAYOUT_KIND=='a' and query_group==1
        from .reader_split import scalar_kernel,vector_kernel
        work=torch.empty((b*m,ns,10,shard),device=u.device,dtype=torch.float32)
        uu,cc,aa,mm,dd,zz,ll,mask0,map0=args
        first=scalar_kernel[(b*m,ns)](uu,cc,aa,zz,ll,mask0,map0,work,logs,cand,t,m,p,candidates,shard,ns,num_warps=warps,enable_fp_fusion=fma)
        vc=cc[...,:16].contiguous() if VECTOR_PACK else cc
        compiled=vector_kernel[(b*m,ns,16//VECTOR_WIDTH)](vc,mm,dd,mask0,map0,work,state,cand,t,m,p,candidates,shard,ns,VD=VECTOR_WIDTH,VP=VECTOR_PACK,num_warps=VECTOR_WARPS,enable_fp_fusion=fma,**({"maxnreg":VECTOR_MAXNREG} if VECTOR_MAXNREG else {}))
    elif query_group==1:
        compiled=kernel[(b*m,ns)](*args,state,logs,cand,t,m,p,candidates,shard,ns,aggregate_float,num_warps=warps,enable_fp_fusion=fma,**launch)
    else:
        assert aggregate_float
        from .reader_group import grouped_reader_kernel
        compiled=grouped_reader_kernel[(b*m//query_group,ns)](*args,state,logs,cand,t,m,p,candidates,shard,ns,query_group,num_warps=warps,enable_fp_fusion=fma)
    KERNEL_STATS[(t,p)]=dict(registers=compiled.n_regs,spills=compiled.n_spills,warps=warps,shard=shard,aggregate_float=aggregate_float,query_group=query_group,fma=fma)
    if SPLIT:KERNEL_STATS[(t,p)].update(vector_maxnreg=VECTOR_MAXNREG,vector_width=VECTOR_WIDTH,vector_warps=VECTOR_WARPS,scalar_registers=first.n_regs,scalar_spills=first.n_spills,intermediate_bytes=work.numel()*work.element_size())
    return state,logs,cand

@reader_reduce.register_fake
def _(u,co,A,mean,detail,z,lm,mask,mapping,candidates,shard,warps,aggregate_float,query_group,fma):
    h,t,m,p=z.shape[:4];b=h*t
    return (u.new_empty((b,m,tr.cdiv(p,shard),3,16),dtype=torch.float64),u.new_empty((b,m,tr.cdiv(p,shard),4),dtype=torch.float64),u.new_empty((b,m,candidates,34),dtype=torch.float64))

def compact_scores(u,en,z,lm,mask,remove,add):
    h,t,m,p=z.shape[:4];b=h*t;r=remove.shape[-1];a=add.shape[-1]
    mask=mask.reshape(b,p);remove=remove.reshape(b,r);add=add.reshape(b,a)
    mapping=torch.full_like(mask,-1,dtype=torch.int32)
    mapping=mapping.scatter(1,remove,torch.arange(r,device=z.device,dtype=torch.int32)[None].expand(b,-1))
    mapping=mapping.scatter(1,add,torch.arange(r,r+a,device=z.device,dtype=torch.int32)[None].expand(b,-1))
    state,logs,cand=reader_reduce(u,en['co'],en['A'],en['mean'],en['detail'],z,lm,mask,mapping,r+a,SHARD,WARPS,AGG32,GROUP,FMA)
    weights=logs[...,:3].softmax(2)
    merged=(weights[...,None]*state).sum(2)
    H=merged[:,:,0];E=merged[:,:,1]-H;ob=merged[:,:,2]
    lb=logs[:,:,:,2].logsumexp(2);maximum=logs[:,:,:,3].amax(2)
    lr,br=cand[:,:,:r,0],cand[:,:,:r,1];la,ba=cand[:,:,r:,0],cand[:,:,r:,1]
    cr,pr=cand[:,:,:r,2:18],cand[:,:,:r,18:];ca,pa=cand[:,:,r:,2:18],cand[:,:,r:,18:]
    er=lr+br;ea=la+ba
    from . import scores_moments
    if MOMENTS and scores_moments.LAZY:
        from .scores import exclude_logmass
        rx=exclude_logmass(er);ax=exclude_logmass(la)
        ro=torch.empty(0,device=er.device,dtype=er.dtype);ao=ro
    else:
        rx,ro=exclude_group(er,cr+pr);ax,ao=exclude_group(la,ca)
    if MOMENTS and scores_moments.FUSED_DEN:
        den=torch.empty((b,m,r,a),device=lb.device,dtype=lb.dtype)
    else:
        if scores_moments.PRECOMPUTE_DEN:
            from .scores_den import first_den
            den=first_den(lb,rx)[...,None]
        else:den=torch.logaddexp(lb[:,:,None,None],rx[:,:,:,None])
        den=torch.logaddexp(den,ax[:,:,None,:])
        den=torch.logaddexp(den,lr[:,:,:,None]);den=torch.logaddexp(den,ea[:,:,None,:])
    if MOMENTS:
        from .scores_moments import moments
        return moments(den,torch.stack((H,E,ob),2),cand,lb,rx,ro,ax,ao,maximum)[0]
    if PACKED:
        from .scores_packed import packed_reduce
        return packed_reduce(den,torch.stack((H,E,ob),2),cand,lb,rx,ro,ax,ao,maximum)
    return action_reduce(den,H,E,lr,er,br,cr,pr,la,ea,ba,ca,pa,lb,ob,rx,ro,ax,ao,maximum)
