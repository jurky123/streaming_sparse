"""Fixed K pages; learned soft KV prototypes; selected-set-dependent swaps.

Related source: migration/unified_router/opt30/model.py (KV mixtures) and
opt32/actions.py (conditional KEEP/SWAP). This is a new small pilot, not a
restoration of unavailable Opt30/32 trained weights.
"""
import math
import torch
from torch import nn


@torch.no_grad()
def prepare(k,v):
    h,n,d=k.shape;k=k.float().reshape(h,n//64,64,d);v=v.float().reshape_as(k)
    dk=k-k.mean(2,keepdim=True);dv=v-v.mean(2,keepdim=True)
    x=torch.cat((dk/dk.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-5),
                 dv/dv.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-5)),-1)
    var=dk.square().mean(2);axis=var.argmax(-1)
    projection=dk.gather(-1,axis[:,:,None,None].expand(h,n//64,64,1)).squeeze(-1)
    projection=projection/projection.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5)
    centers=torch.tensor([-1.5,-.5,.5,1.5],device=k.device)
    anchor=-2*(projection[:,:,:,None]-centers).square()
    return dict(k=k,v=v,x=x,anchor=anchor)


class Prototype(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Embedding(30,8);self.head=nn.Embedding(12,8)
        self.net=nn.Sequential(nn.Linear(272,64),nn.SiLU(),nn.Linear(64,4))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def membership(self,state,layer,heads,learned=True):
        x=state['x'];h,p,n,_=x.shape;score=state['anchor']
        if learned:
            emb=torch.cat((self.layer.weight[layer].expand(h,-1),self.head.weight[heads]),-1)
            emb=emb[:,None,None].expand(h,p,n,16)
            score=score+3*torch.tanh(self.net(torch.cat((x,emb),-1))/3)
        return score.softmax(-1)

    def forward(self,state,layer,heads,learned=True):
        return summarize(state,self.membership(state,layer,heads,learned))


def summarize(state,assignment):
    count=assignment.sum(2).clamp_min(1e-8)
    pk=torch.einsum('hpnr,hpnd->hprd',assignment,state['k'])/count[:,:,:,None]
    pv=torch.einsum('hpnr,hpnd->hprd',assignment,state['v'])/count[:,:,:,None]
    return pk,pv,count


def predict(q,protos,ck,cv):
    pk,pv,count=protos;h,p,r,d=pk.shape
    logits=(q@pk.flatten(1,2).transpose(1,2)/math.sqrt(d)).reshape(h,q.shape[1],p,r)+count.log()[:,None]
    logz=logits.logsumexp(-1)
    values=torch.einsum('hqpr,hprd->hqpd',logits.softmax(-1),pv)
    # Current noisy KV are not preprocessed by the heavy encoder: cheap means
    # are recomputed each step. Their exact attention is retained by the kernel.
    cp=ck.float().reshape(h,-1,64,d).mean(2);vp=cv.float().reshape(h,-1,64,d).mean(2)
    cl=q@cp.transpose(1,2)/math.sqrt(d)+math.log(64)
    cz=cl.logsumexp(-1);cval=cl.softmax(-1)@vp
    z=torch.cat((logz,cz[:,:,None]),-1);val=torch.cat((values,cval[:,:,None]),2)
    output=(z.softmax(-1)[:,:,:,None]*val).sum(2)
    return z,val,output


def initial_mask(logz,ratio=.2):
    p=logz.shape[-1]-1;keep=max(1,int(p*ratio))
    mask=torch.zeros_like(logz,dtype=torch.bool)
    mask[...,:p].scatter_(-1,logz[...,:p].topk(keep,dim=-1).indices,True)
    mask[...,-1]=True
    return mask


def output_for_masks(logz,values,masks):
    weights=logz[:,:,None,:].masked_fill(~masks,-torch.inf).softmax(-1)
    return torch.einsum('hqap,hqpd->hqad',weights,values)


def alternatives(logz,values,mask):
    """KEEP + 2 weakest selected x 2 strongest unselected; one equal-cost swap."""
    p=logz.shape[-1]-1
    remove=logz[...,:p].masked_fill(~mask[...,:p],torch.inf).topk(2,largest=False,dim=-1).indices
    add=logz[...,:p].masked_fill(mask[...,:p],-torch.inf).topk(2,dim=-1).indices
    ri=remove.repeat_interleave(2,-1);aj=add.repeat(1,1,2)
    page=torch.arange(p+1,device=logz.device)
    swapped=(mask[:,:,None,:]&(page!=ri[:,:,:,None]))|(page==aj[:,:,:,None])
    masks=torch.cat((mask[:,:,None],swapped),2)
    outputs=output_for_masks(logz,values,masks)
    full=(logz.softmax(-1)[:,:,:,None]*values).sum(2)
    scale=full.square().mean(-1).clamp_min(1e-5)
    errors=(outputs-full[:,:,None]).square().mean(-1)/scale[:,:,None]
    gains=errors[:,:,:1]-errors
    prob=logz.softmax(-1);z=(prob*mask).sum(-1)
    delta=outputs-outputs[:,:,:1]
    mi=prob.gather(-1,ri);mj=prob.gather(-1,aj)
    zeros=torch.zeros_like(z[:,:,None])
    mi=torch.cat((zeros,mi),-1);mj=torch.cat((zeros,mj),-1)
    constant=lambda a:a[:,:,None].expand_as(gains)
    features=torch.stack((gains,errors,delta.square().mean(-1)/scale[:,:,None],
        constant(z),mi,mj,mi/z[:,:,None].clamp_min(1e-5),mj/z[:,:,None].clamp_min(1e-5),
        constant(errors[:,:,0]),constant(prob[...,-1]),
        constant((-(prob*prob.clamp_min(1e-20).log()).sum(-1))/math.log(p+1)),
        constant(full.norm(dim=-1).log1p())),dim=-1)
    assert torch.equal(masks.sum(-1),mask.sum(-1)[:,:,None].expand_as(gains))
    return masks,features,gains


class Corrector(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features,gains):
        score=gains+.05*self.net(features.clamp(-20,20)).squeeze(-1)
        return score-score[:,:,:1]


def choose(masks,scores):
    action=scores.argmax(-1)
    chosen=masks.gather(2,action[:,:,None,None].expand(-1,-1,1,masks.shape[-1])).squeeze(2)
    return chosen,action
