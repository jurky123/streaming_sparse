"""FP64 per-action reduction with query-shared coefficients and conditional fallback."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .scores import gather_pages,exclude_group

@tr.jit
def action_kernel(DEN,H,E,LR,ER,BR,CR,PR,LA,EA,BA,CA,PA,LB,OB,RX,RO,AX,AO,MAX,OUT,R:tl.constexpr,A:tl.constexpr,M:tl.constexpr,D:tl.constexpr):
    action=tl.program_id(0);batch=action//(R*A);ri=(action//A)%R;ai=action%A
    m=tl.arange(0,M);d=tl.arange(0,D)
    den=tl.load(DEN+((batch*M+m)*R+ri)*A+ai)
    hr=tl.load(H+(batch*M+m[:,None])*D+d[None,:]);err=tl.load(E+(batch*M+m[:,None])*D+d[None,:])
    lr=tl.load(LR+(batch*M+m)*R+ri);er=tl.load(ER+(batch*M+m)*R+ri);br=tl.load(BR+(batch*M+m)*R+ri)
    la=tl.load(LA+(batch*M+m)*A+ai);ea=tl.load(EA+(batch*M+m)*A+ai);ba=tl.load(BA+(batch*M+m)*A+ai)
    cr=tl.load(CR+((batch*M+m[:,None])*R+ri)*D+d[None,:]);pr=tl.load(PR+((batch*M+m[:,None])*R+ri)*D+d[None,:])
    ca=tl.load(CA+((batch*M+m[:,None])*A+ai)*D+d[None,:]);pa=tl.load(PA+((batch*M+m[:,None])*A+ai)*D+d[None,:])
    xer=lib.exp(er-den);xar=lib.exp(lr-den);xea=lib.exp(ea-den);xaa=lib.exp(la-den)
    dr=tl.where(tl.abs(br)<.5,xar*lib.expm1(tl.minimum(tl.maximum(br,-.5),.5)),xer-xar)
    da=tl.where(tl.abs(ba)<.5,xaa*lib.expm1(tl.minimum(tl.maximum(ba,-.5),.5)),xea-xaa)
    delta=da[:,None]*(ca-hr)+xea[:,None]*pa-(dr[:,None]*(cr-hr)+xer[:,None]*pr)
    maximum=tl.load(MAX+batch*M+m)
    finite=(tl.abs(delta)<float('inf'))
    bad=(maximum-den>80)|(tl.sum(finite.to(tl.int32),1)!=D)
    if tl.sum(bad.to(tl.int32),0)>0:
        lb=tl.load(LB+batch*M+m);rx=tl.load(RX+(batch*M+m)*R+ri);ax=tl.load(AX+(batch*M+m)*A+ai)
        ob=tl.load(OB+(batch*M+m[:,None])*D+d[None,:]);ro=tl.load(RO+((batch*M+m[:,None])*R+ri)*D+d[None,:]);ao=tl.load(AO+((batch*M+m[:,None])*A+ai)*D+d[None,:])
        new=lib.exp(lb-den)[:,None]*ob+lib.exp(rx-den)[:,None]*ro
        new=new+lib.exp(ax-den)[:,None]*ao
        new=new+xar[:,None]*cr
        new=new+xea[:,None]*(ca+pa)
        delta=tl.where(bad[:,None],new-hr,delta)
    gain=2*tl.sum(err*delta,1)-tl.sum(delta*delta,1)
    tl.store(OUT+action,tl.sum(gain,0)/M)

@torch.library.custom_op('streaming_sparse::action_reduce',mutates_args=())
def action_reduce(den:torch.Tensor,h:torch.Tensor,e:torch.Tensor,lr:torch.Tensor,er:torch.Tensor,br:torch.Tensor,cr:torch.Tensor,pr:torch.Tensor,la:torch.Tensor,ea:torch.Tensor,ba:torch.Tensor,ca:torch.Tensor,pa:torch.Tensor,lb:torch.Tensor,ob:torch.Tensor,rx:torch.Tensor,ro:torch.Tensor,ax:torch.Tensor,ao:torch.Tensor,maximum:torch.Tensor)->torch.Tensor:
    b,m,r,a=den.shape;d=h.shape[-1];out=torch.empty((b,r,a),device=den.device,dtype=torch.float64)
    args=[x.contiguous() for x in (den,h,e,lr,er,br,cr,pr,la,ea,ba,ca,pa,lb,ob,rx,ro,ax,ao,maximum)]
    action_kernel[(b*r*a,)](*args,out,r,a,m,d,num_warps=4,enable_fp_fusion=False)
    return out

@action_reduce.register_fake
def _(den,h,e,lr,er,br,cr,pr,la,ea,ba,ca,pa,lb,ob,rx,ro,ax,ao,maximum):
    return den.new_empty((den.shape[0],den.shape[2],den.shape[3]))

@torch.no_grad()
def grouped_action_scores(rp,beta,lm,center,mask,remove,add,background=False,factored=False,moment=False):
    rp,beta,lm,center=(x.double() for x in (rp,beta,lm,center))
    le=lm+beta;v=center+rp;logs=torch.where(mask[:,None],le,lm)
    candidates=torch.zeros_like(mask).scatter(1,remove,True).scatter(1,add,True)
    if background:
        from .score_background import score_background
        H,E,lb,ob,maximum=score_background(rp,beta,lm,center,mask,candidates)
    else:
        H=(logs.softmax(-1)[...,None]*(center+mask[:,None,:,None]*rp)).sum(2)
        target=(le.softmax(-1)[...,None]*v).sum(2);E=target-H
        bg=logs.masked_fill(candidates[:,None],-torch.inf)
        lb=bg.logsumexp(-1);ob=(bg.softmax(-1)[...,None]*(center+mask[:,None,:,None]*rp)).sum(2)
        maximum=logs.amax(-1)
    lr=gather_pages(lm,remove);er=gather_pages(le,remove);cr=gather_pages(center,remove);pr=gather_pages(rp,remove)
    la=gather_pages(lm,add);ea=gather_pages(le,add);ca=gather_pages(center,add);pa=gather_pages(rp,add)
    from . import scores_moments
    if moment and scores_moments.LAZY:
        from .scores import exclude_logmass
        rx=exclude_logmass(er);ax=exclude_logmass(la)
        ro=torch.empty(0,device=er.device,dtype=er.dtype);ao=ro
    else:
        rx,ro=exclude_group(er,cr+pr);ax,ao=exclude_group(la,ca)
    den=torch.logaddexp(lb[:,:,None,None],rx[:,:,:,None]);den=torch.logaddexp(den,ax[:,:,None,:]);den=torch.logaddexp(den,lr[:,:,:,None]);den=torch.logaddexp(den,ea[:,:,None,:])
    if moment:
        from .scores_moments import moments
        cand=torch.cat((torch.cat((lr[...,None],gather_pages(beta,remove)[...,None],cr,pr),-1),torch.cat((la[...,None],gather_pages(beta,add)[...,None],ca,pa),-1)),2)
        return moments(den,torch.stack((H,E,ob),2),cand,lb,rx,ro,ax,ao,maximum)[0]
    reducer=action_reduce
    if factored:
        from .scores_factored import reduce
        reducer=reduce
    return reducer(den,H,E,lr,er,gather_pages(beta,remove),cr,pr,la,ea,gather_pages(beta,add),ca,pa,lb,ob,rx,ro,ax,ao,maximum)

fused_action_scores=torch.compile(grouped_action_scores,fullgraph=True,options={'triton.cudagraphs':False})


def background_action_scores(rp,beta,lm,center,mask,remove,add):
    return grouped_action_scores(rp,beta,lm,center,mask,remove,add,background=True)

fused_background_scores=torch.compile(background_action_scores,fullgraph=True,options={"triton.cudagraphs":False})


def factored_action_scores(rp,beta,lm,center,mask,remove,add):
    return grouped_action_scores(rp,beta,lm,center,mask,remove,add,factored=True)

fused_factored_scores=torch.compile(factored_action_scores,fullgraph=True,options={"triton.cudagraphs":False})


def moment_action_scores(rp,beta,lm,center,mask,remove,add):
    return grouped_action_scores(rp,beta,lm,center,mask,remove,add,moment=True)

fused_moment_scores=torch.compile(moment_action_scores,fullgraph=True,options={"triton.cudagraphs":False})
