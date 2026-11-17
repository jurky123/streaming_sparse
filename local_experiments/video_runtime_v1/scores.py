"""FP64 shared background scoring. No Dense inputs and no full-page action axis."""
import torch
GROUP_BMM=False
FUSED_EXCLUDE=False
SCAN_EXCLUDE=False

def gather_pages(x,ids):
    # x B,M,P[,D], ids B,K
    index=ids[:,None].expand(-1,x.shape[1],-1)
    if x.ndim==4:index=index[...,None].expand(-1,-1,-1,x.shape[-1])
    return x.gather(2,index)

def exclude_group(logs,values):
    if SCAN_EXCLUDE:
        from .exclude_scan import exclude_scan
        return exclude_scan(logs,values)
    if FUSED_EXCLUDE:
        from .exclude_fused import exclude_fused
        return exclude_fused(logs,values)
    k=logs.shape[-1]
    eye=torch.eye(k,device=logs.device,dtype=torch.bool)
    excluded=logs[:,:,None,:].expand(-1,-1,k,-1).masked_fill(eye,-torch.inf)
    if GROUP_BMM:
        from .group_bmm import group_bmm
        return excluded.logsumexp(-1),group_bmm(excluded.softmax(-1),values)
    return excluded.logsumexp(-1),excluded.softmax(-1)@values

def exclude_logmass(logs):
    k=logs.shape[-1]
    eye=torch.eye(k,device=logs.device,dtype=torch.bool)
    return logs[:,:,None,:].expand(-1,-1,k,-1).masked_fill(eye,-torch.inf).logsumexp(-1)

@torch.no_grad()
def grouped_scores(rp,beta,lm,center,mask,remove,add):
    """B,M,P,D inputs; Cartesian R x A pool, return B,R,A raw proxy gains.

    Positive five-group denominators survive removing the dominant page.
    Endpoint deltas preserve tiny gains. If endpoint cancellation is unsafe,
    reconstruct from independently normalized groups, never subtract masses.
    """
    rp,beta,lm,center=(x.double() for x in (rp,beta,lm,center))
    le=lm+beta;v=center+rp;logs=torch.where(mask[:,None],le,lm)
    H=(logs.softmax(-1)[...,None]*(center+mask[:,None,:,None]*rp)).sum(2)
    target=(le.softmax(-1)[...,None]*v).sum(2);E=target-H
    candidates=torch.zeros_like(mask).scatter(1,remove,True).scatter(1,add,True)
    bg=logs.masked_fill(candidates[:,None],-torch.inf)
    lb=bg.logsumexp(-1);ob=(bg.softmax(-1)[...,None]*(center+mask[:,None,:,None]*rp)).sum(2)
    lr=gather_pages(lm,remove);er=gather_pages(le,remove);cr=gather_pages(center,remove);pr=gather_pages(rp,remove)
    la=gather_pages(lm,add);ea=gather_pages(le,add);ca=gather_pages(center,add);pa=gather_pages(rp,add)
    rx,ro=exclude_group(er,cr+pr);ax,ao=exclude_group(la,ca)
    den=torch.logaddexp(lb[:,:,None,None],rx[:,:,:,None])
    den=torch.logaddexp(den,ax[:,:,None,:]);den=torch.logaddexp(den,lr[:,:,:,None]);den=torch.logaddexp(den,ea[:,:,None,:])
    # Positive mixtures only. Their vector aggregates are shared over all actions.
    new=(lb[:,:,None,None]-den).exp()[...,None]*ob[:,:,None,None]
    new=new+(rx[:,:,:,None]-den).exp()[...,None]*ro[:,:,:,None]
    new=new+(ax[:,:,None,:]-den).exp()[...,None]*ao[:,:,None]
    new=new+(lr[:,:,:,None]-den).exp()[...,None]*cr[:,:,:,None]
    new=new+(ea[:,:,None,:]-den).exp()[...,None]*(ca+pa)[:,:,None]
    # Endpoint differences; expm1 keeps beta close to zero distinguishable.
    def endpoint(l,e,c,p,ids,axis):
        bb=gather_pages(beta,ids)
        if axis=='r':l,e,c,p,bb=l[:,:,:,None],e[:,:,:,None],c[:,:,:,None],p[:,:,:,None],bb[:,:,:,None]
        else:l,e,c,p,bb=l[:,:,None,:],e[:,:,None,:],c[:,:,None],p[:,:,None],bb[:,:,None,:]
        exact=(e-den).exp();approx=(l-den).exp()
        difference=torch.where(bb.abs()<.5,approx*torch.expm1(bb.clamp(-.5,.5)),exact-approx)
        return difference[...,None]*(c-H[:,:,None,None])+exact[...,None]*p
    delta=endpoint(la,ea,ca,pa,add,'a')-endpoint(lr,er,cr,pr,remove,'r')
    bad=(logs.amax(-1)[:,:,None,None]-den>80)|(~torch.isfinite(delta).all(-1))
    delta=torch.where(bad[...,None],new-H[:,:,None,None],delta)
    return (2*(E[:,:,None,None]*delta).sum(-1)-delta.square().sum(-1)).mean(1)

@torch.no_grad()
def candidate_pool(risk,mask,own):
    """Original stable low-risk 16 removes / high-risk 32 adds, on GPU."""
    k=min(16,int(mask.shape[-1]*.1)-1)
    removable=mask.scatter(-1,own[...,None],False)
    r=risk.masked_fill(~removable,torch.inf).argsort(dim=-1,stable=True)[...,:k]
    a=risk.masked_fill(mask,-torch.inf).argsort(dim=-1,descending=True,stable=True)[...,:32]
    return r,a

@torch.no_grad()
def apply_best(mask,remove,add,scores):
    b=mask.shape[0];flat=scores.flatten(1);gain,j=flat.max(-1);execute=(gain>0)&torch.isfinite(flat).all(-1)
    rr=remove.gather(1,(j//add.shape[-1])[:,None]);aa=add.gather(1,(j%add.shape[-1])[:,None])
    result=mask.scatter(1,rr,~execute[:,None]);result=result.scatter(1,aa,execute[:,None])
    return result,gain,execute

# Fuse FP64 pointwise action arithmetic and reductions; retain eager reference.
# CUDA Graph ownership remains with BatchGraphs, which warms each shape first.
fused_grouped_scores = torch.compile(grouped_scores, fullgraph=True,
                                     options={'triton.cudagraphs': False})
