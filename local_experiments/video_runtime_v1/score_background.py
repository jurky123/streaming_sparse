"""FP64 shared page mixtures without three expanded weighted-value tensors."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

@tr.jit
def background_kernel(RP,BETA,LM,C,MASK,EXCLUDED,H,E,LB,OB,MAX,
                      P:tl.constexpr,M:tl.constexpr,D:tl.constexpr,S:tl.constexpr):
    row=tl.program_id(0);b=row//M
    p=tl.arange(0,S);d=tl.arange(0,D)
    beta=tl.load(BETA+row*P+p,p<P,0).to(tl.float64)
    lm=tl.load(LM+row*P+p,p<P,-float('inf')).to(tl.float64)
    mask=tl.load(MASK+b*P+p,p<P,0)
    excluded=tl.load(EXCLUDED+b*P+p,p<P,1)
    logs=tl.where(mask,lm+beta,lm);le=lm+beta
    mx=tl.max(logs,0);me=tl.max(le,0);bg=tl.where(excluded,-float('inf'),logs);mb=tl.max(bg,0)
    w=lib.exp(logs-mx);w=w/tl.sum(w,0)
    we=lib.exp(le-me);we=we/tl.sum(we,0)
    wb=lib.exp(bg-mb);zb=tl.sum(wb,0);wb=wb/zb
    c=tl.load(C+(row*P+p[:,None])*D+d[None,:],p[:,None]<P,0).to(tl.float64)
    rp=tl.load(RP+(row*P+p[:,None])*D+d[None,:],p[:,None]<P,0).to(tl.float64)
    current=c+tl.where(mask[:,None],rp,0.)
    h=tl.sum(w[:,None]*current,0);target=tl.sum(we[:,None]*(c+rp),0);ob=tl.sum(wb[:,None]*current,0)
    tl.store(H+row*D+d,h);tl.store(E+row*D+d,target-h);tl.store(OB+row*D+d,ob)
    tl.store(LB+row,mb+lib.log(zb));tl.store(MAX+row,mx)

@torch.library.custom_op('streaming_sparse::score_background',mutates_args=())
def score_background(rp:torch.Tensor,beta:torch.Tensor,lm:torch.Tensor,center:torch.Tensor,mask:torch.Tensor,excluded:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    rp,beta,lm,center,mask,excluded=(x.contiguous() for x in (rp,beta,lm,center,mask,excluded))
    b,m,p,d=rp.shape
    h=torch.empty((b,m,d),device=rp.device,dtype=torch.float64);e=torch.empty_like(h);ob=torch.empty_like(h)
    lb=torch.empty((b,m),device=rp.device,dtype=torch.float64);maximum=torch.empty_like(lb)
    background_kernel[(b*m,)](rp,beta,lm,center,mask,excluded,h,e,lb,ob,maximum,p,m,d,tr.next_power_of_2(p),num_warps=8,enable_fp_fusion=False)
    return h,e,lb,ob,maximum

@score_background.register_fake
def _(rp,beta,lm,center,mask,excluded):
    b,m,p,d=rp.shape
    v=lambda:rp.new_empty((b,m,d),dtype=torch.float64)
    s=lambda:rp.new_empty((b,m),dtype=torch.float64)
    return v(),v(),s(),v(),s()
