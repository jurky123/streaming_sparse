"""Frozen-input local objective; no network, QKV or oracle-score gradients."""
import math
import torch
import torch.nn.functional as F
from streaming_sparse.v08.attention import costs, masks


def page_statistics(q,kh,vh,kc,vc):
    """H,Q,D inputs; stable true original-KV page attention, not prototypes."""
    h,n,d=kh.shape;b=math.ceil(n/64);q=q.float()
    # One canonical raw-token GEMM; splitting K changes FP32 reduction kernels
    # at large real logits. Keep Router's current-only GEMM separate below.
    all_logits=torch.bmm(q,torch.cat((kh,kc),1).float().transpose(1,2))/math.sqrt(d)
    lh=all_logits[...,:n];attention_lc=all_logits[...,n:]
    lc=torch.bmm(q,kc.float().transpose(1,2))/math.sqrt(d)
    lp=F.pad(lh,(0,b*64-n),value=-torch.inf).reshape(h,len(q[0]),b,64).double()
    lz=lp.logsumexp(-1)
    vv=F.pad(vh.float(),(0,0,0,b*64-n)).reshape(h,b,64,d)
    mu=torch.einsum('hqbt,hbtd->hqbd',lp.softmax(-1),vv.double()).float()
    # Router's current representation retains its original FP32 semantics.
    current_w=lc.softmax(-1)
    return dict(page_logz=lz,page_value=mu,current_logz=lc.logsumexp(-1),
        current_key=torch.bmm(current_w,kc.float()),current_value=torch.bmm(current_w,vc.float()),
        exact_current_logz=attention_lc.double().logsumexp(-1),
        exact_current_value=torch.bmm(attention_lc.double().softmax(-1),vc.double()).float())


def output(data,mask):
    """Exact page algebra, FP64 reductions; differentiable only in the gate."""
    lz=data['page_logz'];lc=data['exact_current_logz']
    # Shift must include removed pages: matches the original continuous VJP.
    shift=torch.maximum(lz.amax(-1),lc).detach()
    z=(lz-shift[...,None]).exp();zc=(lc-shift).exp()
    weights=z*mask.double();den=zc+weights.sum(-1)
    num=torch.einsum('hqb,hqbd->hqd',weights,data['page_value'].double())
    num=num+zc[...,None]*data['exact_current_value'].double()
    return (num/den[...,None]).float()


def linear(x,modules,pairs,index=None):
    seq=[modules[i] if index is None else modules[i][index] for i in pairs]
    weight=torch.stack([m.weight for m in seq]);bias=torch.stack([m.bias for m in seq])
    return torch.bmm(x.flatten(1,-2),weight.transpose(1,2)).reshape(*x.shape[:-1],weight.shape[1])+bias.reshape(len(pairs),*([1]*(x.ndim-2)),-1)


def score(router,data,layer,step,features_only=False):
    """Batch 12 independent heads; same original P8/full128 Router equations."""
    pairs=list(range(12))if router.sketch.pairs==12 else list(range((layer-1)*12,layer*12))
    q=data['q'].float();kh=data['kh'].float()
    h,n,d=kh.shape;b=math.ceil(n/64);kk=F.pad(kh,(0,0,0,b*64-n)).reshape(h,b,64,d)
    valid=torch.arange(b*64,device=q.device).reshape(b,64)<n
    a=linear(kk,router.sketch.assignment,pairs).softmax(-1)*valid[None,...,None]
    count=a.sum(2);den=count.clamp_min(1e-12)[...,None]
    key=torch.einsum('hbtp,hbtd->hbpd',a,kk)/den
    var=(torch.einsum('hbtp,hbtd->hbpd',a,kk.square())/den-key.square()).clamp_min(0)
    logits=torch.bmm(q,key.flatten(1,2).transpose(1,2)).reshape(h,q.shape[1],b,8)/math.sqrt(d)
    good=count>1e-8;logits=(logits+count.clamp_min(1e-30).log()[:,None]).masked_fill(~good[:,None],-torch.inf)
    w=logits.softmax(-1);raw=logits.logsumexp(-1)
    entropy=-(w*w.clamp_min(1e-30).log()).sum(-1)
    qnorm=q.norm(dim=-1).clamp_min(1e-12);direction=q/qnorm[...,None]
    pc=good.sum(-1).clamp_min(1)[:,None];finite=torch.where(good[:,None],logits,torch.zeros_like(logits))
    lm=finite.sum(-1)/pc
    lv=torch.where(good[:,None],(finite-lm[...,None]).square(),torch.zeros_like(logits)).sum(-1)/pc
    # R1 equal-valid-prototype average; no count weighting.
    avgvar=(var*good[...,None]).sum(2)/good.sum(-1).clamp_min(1)[...,None]
    kvar=torch.bmm(direction.square(),avgvar.transpose(1,2))
    c=costs(n,64,q.device)
    features=torch.stack((qnorm.log()[...,None].expand_as(raw),c.log()[None,None].expand_as(raw),
        lv.clamp_min(1e-12).sqrt(),logits.amax(-1)-lm,entropy,kvar),-1)
    if features_only:return features
    norm=(features-router.sketch.feature_mean[pairs,None,None])/router.sketch.feature_std[pairs,None,None].clamp_min(1e-3)
    cal=linear(F.gelu(linear(norm,router.sketch.calibrator.networks,pairs,0)),router.sketch.calibrator.networks,pairs,2).squeeze(-1)
    logz=raw+cal
    # The K-only scoring forward never indexes V, page_value or V statistics.
    content=key
    if router.kind=='V':
        vv=F.pad(data['vh'].float(),(0,0,0,b*64-n)).reshape_as(kk)
        content=torch.einsum('hbtp,hbtd->hbpd',a,vv)/den
    mu=torch.einsum('hqbp,hbpd->hqbd',w,content)
    oc=data['current_key' if router.kind=='K' else 'current_value']
    mass=torch.cat((logz.double(),data['current_logz'].double()[...,None]),-1).softmax(-1)
    mh=mass[...,:-1];mc=mass[...,-1]
    dense=(mh[...,None]*mu.double()).sum(2)+mc[...,None]*oc.double()
    residual=(mh[...,None]*(mu.double()-dense[:,:,None])).float()
    norm_r=residual.norm(dim=-1).clamp_min(1e-12)
    projection=linear(residual,router.content,pairs)
    projection=F.layer_norm(projection,(8,),eps=router.norm[pairs[0]].eps)
    projection=projection*torch.stack([router.norm[i].weight for i in pairs])[:,None,None]+torch.stack([router.norm[i].bias for i in pairs])[:,None,None]
    const=lambda value:torch.full_like(logz,float(value))
    scalars=torch.stack((logz-data['current_logz'][...,None],c.log()[None,None].expand_as(logz),
        const(.2),qnorm.log()[...,None].expand_as(logz),const(step),const(math.log(n)),
        mc.float()[...,None].expand_as(logz),entropy),-1)
    x=torch.cat((scalars,projection,norm_r.log()[...,None]),-1)
    delta=router.scale[pairs,None,None]*linear(F.gelu(linear(x,router.head.networks,pairs,0)),router.head.networks,pairs,2).squeeze(-1)
    return dict(scores=logz+delta,logz=logz,features=features,costs=c)


def loss(router,data,layer,step,hard_only=False,mass_only=False):
    result=score(router,data,layer,step);s=result['logz']if mass_only else result['scores'];shape=s.shape
    if hard_only:
        from streaming_sparse.v08.execution.accelerate import page_select
        mask=page_select(s.flatten(0,1),result['costs'],math.floor(.2*len(data['kh'][0]))).reshape(shape)
    else:mask=masks(s.flatten(0,1),result['costs'],.2)[0].reshape(shape)
    pred=output(data,mask);target=data['target'];diff=(pred-target).square()
    value=diff.sum()/(target.square().sum()+1e-8)
    byhead=diff.sum((1,2))/(target.square().sum((1,2))+1e-8)
    return value,dict(head_nmse=byhead.detach(),actual_keep=((mask.detach()*result['costs']).sum(-1)/len(data['kh'][0])).mean().detach())
