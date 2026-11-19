"""Repair nonfinite legacy expressions with equivalent positive log mixtures.

Finite legacy results are retained. No sample/action is dropped, clipped or
assigned a synthetic target. Stored normalized page teachers recover information
lost when scaled N/Z underflows or dominant masses cancel during subtraction.
"""
import math
import torch
from ..review_v1.ops import haar
from ..e24_joint_representatives.core import gains,objective,predicted
from ..e24_joint_representatives.train import getstate
from ..e12_swap_prediction.core import pool


def pages(c,d,s,m):
    if '_normalized_exact' not in d:d['_normalized_exact']=d['ev']['page_output'].cuda(non_blocking=True).double()
    basis=m.basis[0].bfloat16().double()
    values=s.mean[0].double()[:,None]+torch.einsum('ar,pak,dk->prd',haar(d['q']).double(),s.detail[0].double(),basis)
    approx=torch.einsum('qpr,prd->qpd',c['logit'].softmax(-1),values)
    return d['_normalized_exact'],approx


def mixed_log(c,d,s,m,masks):
    exact,approx=pages(c,d,s,m);logs=torch.where(masks[None],d['logZ'][:,None],c['logzh'][:,None])
    weights=logs.softmax(-1)
    return torch.bmm(weights*masks[None],exact)+torch.bmm(weights*(~masks)[None],approx)


def safe_objective(c,d,s,m,mask):
    old=objective(c,d['N'],d['Z'],d['dense'],mask)
    if bool(torch.isfinite(old)):return old
    output=mixed_log(c,d,s,m,mask[None])[:,0]
    d.setdefault('numeric_repairs',[]).append('objective_log_mixture')
    return (output-d['dense']).square().sum()/(d['dense'].square().sum()+1e-12)


@torch.no_grad()
def true_gains(c,d,s,m,mask,r,a):
    old=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);bad=~torch.isfinite(old)
    if not bool(bad.any()):return old
    rr,aa=r[bad],a[bad];masks=mask[None].expand(len(rr),-1).clone();i=torch.arange(len(rr),device=r.device);masks[i,rr]=False;masks[i,aa]=True
    output=mixed_log(c,d,s,m,masks);base=safe_objective(c,d,s,m,mask)
    replacement=base-(output-d['dense'][:,None]).square().sum((0,2))/(d['dense'].square().sum()+1e-12)
    if not bool(torch.isfinite(replacement).all()):raise FloatingPointError('Log-domain true gains remain nonfinite')
    fixed=old.clone();fixed[bad]=replacement
    d.setdefault('numeric_repairs',[]).append(dict(true_gain_actions=int(bad.sum())))
    return fixed


@torch.no_grad()
def state(c,d,s,m,mask,r,a,mode,risk,rng):
    if mode not in (6,7):return getstate(c,d,mask,r,a,mode,risk,rng)
    for _ in range(1 if mode==6 else 2):
        g=true_gains(c,d,s,m,mask,r,a);j=int(g.argmax())
        if g[j]>0:mask=mask.clone();mask[r[j]]=False;mask[a[j]]=True
        rr,aa=pool({'risk':risk},mask.cpu().numpy(),d['own']);r=torch.as_tensor(rr,device=r.device);a=torch.as_tensor(aa,device=a.device)
    return mask,r,a


def proxy_gains(reader,q,s,cp,mask,r,a,z,uv,beta):
    try:old=predicted(z,uv,cp['mass'],mask,r,a,beta)
    except AssertionError:old=None
    if old is not None and bool(torch.isfinite(old).all()):
        # A subtraction denominator can be tiny positive yet catastrophically wrong.
        # Compare it with a positive sum before trusting a finite proxy gain.
        with torch.no_grad():
            mass=cp['mass'];exact=mass*beta.double().exp();current=torch.where(mask[None],exact,mass).sum(1)
            ms=mask[None].expand(len(r),-1).clone();ii=torch.arange(len(r),device=q.device);ms[ii,r]=False;ms[ii,a]=True
            mf=ms.double();positive=exact@mf.T+mass@(1-mf).T
            subtractive=current[:,None]-uv[:,r]+uv[:,a]
            tolerance=math.sqrt(torch.finfo(torch.float64).eps)
            unsafe=(~torch.isfinite(positive))|(positive<=0)|((subtractive-positive).abs()>tolerance*positive)
        if not bool(unsafe.any()):return old,False
    # Recover the pre-mass neural response rather than divide z by an underflowed mass.
    parts=[s.keys[0].float().flatten(1),s.mean[0].float(),s.detail[0].float().flatten(1),s.stats[0].float().flatten(1)]
    features=torch.cat([p/(p.detach().square().mean(-1,keepdim=True).sqrt()+1e-4) for p in parts],1)
    co=reader.page(features).reshape(-1,5,17).bfloat16().float();relative=cp['logit']-cp['logit'].mean(-1,keepdim=True)
    phi=torch.tanh(reader.query(q.float())[:,None]/math.sqrt(128)+relative).to(co.dtype)
    value=co[None,:,0]+torch.einsum('qpj,pjk->qpk',phi,co[:,1:]);response=value[...,:16].double()
    lm=cp['logzh']-cp['A'].log()[:,None];le=lm+value[...,16].double()
    target=(torch.exp(lm-le.logsumexp(1)[:,None])[...,None]*response).sum(1)
    current=torch.where(mask[None],le,lm).logsumexp(1)
    H=(torch.where(mask[None],lm-current[:,None],-torch.inf).exp()[...,None]*response).sum(1);E=target-H
    scores=[]
    for start in range(0,len(r),64):
        rr,aa=r[start:start+64],a[start:start+64];ms=mask[None].expand(len(rr),-1).clone();i=torch.arange(len(rr),device=q.device);ms[i,rr]=False;ms[i,aa]=True
        den=torch.where(ms[None],le[:,None],lm[:,None]).logsumexp(-1)
        weights=torch.where(ms[None],lm[:,None]-den[...,None],-torch.inf).exp()
        new=torch.bmm(weights,response);delta=new-H[:,None]
        scores.append((2*(E[:,None]*delta).sum(-1)-delta.square().sum(-1)).mean(0))
    result=torch.cat(scores)
    if not bool(torch.isfinite(result).all()):raise FloatingPointError('Log-domain proxy remains nonfinite')
    return result,True
