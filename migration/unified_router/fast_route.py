"""Batched routing, cached projections, fused candidates and compiled features."""
import math
import torch
from .core import features
from .fast_kernels import candidate_sides
from .learning import choose


def prepare(sketch,kind):
    basis=sketch.key_basis[...,:64]
    projected=[]
    for p in sketch.parts:
        h,b,protos,_=p.key.shape
        key=torch.bmm(p.key.flatten(1,2),basis)
        value=None
        if kind!='U-Sol':
            source=p.key if kind=='K32-Set' else p.value
            content_basis=sketch.key_basis[...,:32] if kind=='K32-Set' else sketch.value_basis[...,:32]
            value=torch.bmm(source.flatten(1,2),content_basis).reshape(h,b,protos,32)
        logcount=p.count.clamp_min(1e-30).log().masked_fill(p.count<=1e-8,-torch.inf)
        projected.append((key,logcount,value))
    return basis,projected


def score_cached(q,basis,projected):
    qq=torch.bmm(q.float(),basis)
    score=[];content=[]
    for key,logcount,value in projected:
        h,b,p=logcount.shape
        ll=torch.bmm(qq,key.transpose(1,2)).reshape(h,q.shape[1],b,p)/math.sqrt(128)+logcount[:,None]
        score.append(ll.logsumexp(-1))
        if value is not None:
            content.append(torch.einsum('hqbp,hbpr->hqbr',ll.softmax(-1),value))
    return torch.cat(score,2),torch.cat(content,2) if content else None


def mask_only(score,beta):
    std,mean=torch.std_mean(score,dim=-1,unbiased=False,keepdim=True)
    mask=(score>mean+beta*std)|(std<=1e-6)
    idx=score.argmax(-1,keepdim=True)
    return mask.scatter(-1,idx,mask.gather(-1,idx)|(~mask.any(-1,keepdim=True)))


def action_dict(rem,add,mask):
    n=rem.shape[0]
    absent=torch.full((n,1),-1,device=mask.device,dtype=torch.long)
    i=torch.cat((absent,absent.expand(-1,8),rem,rem.repeat_interleave(8,-1)),-1)
    j=torch.cat((absent,add,absent.expand(-1,8),add.repeat(1,8)),-1)
    valid=torch.cat((torch.ones(n,1,device=mask.device,dtype=torch.bool),add>=0,
        (rem>=0)&(mask.flatten(0,1).sum(-1,keepdim=True)>1),
        (rem.repeat_interleave(8,-1)>=0)&(add.repeat(1,8)>=0)),-1)
    return dict(remove=i,add=j,valid=valid)


def finish(q,sketch,score,mask,rem,add,beta,step,value,head):
    actions=action_dict(rem,add,mask)
    geometry,key=features(q,sketch,score,mask,actions,beta,0,value)
    key=torch.cat((key[...,:8],(step/3).expand_as(key[...,8:9]),key[...,9:]),-1)
    x=torch.cat((geometry,key),-1).reshape(12,q.shape[1],81,24)
    pred=head(x)
    selected=choose(pred,actions['valid'].reshape_as(pred),torch.zeros_like(pred),0.).flatten()
    i=actions['remove'].gather(-1,selected[:,None]).reshape(mask.shape[:2])
    j=actions['add'].gather(-1,selected[:,None]).reshape(mask.shape[:2])
    page=torch.arange(mask.shape[-1],device=mask.device)
    return (mask&(page!=i[...,None]))|(page==j[...,None])


class FastRoute:
    def __init__(self,compile=True,fused=False,query_stride=1,batch=512):
        self.query_stride=query_stride;self.batch=batch
        # Fusion removes large temporary feature tensors and individual pointwise
        # launches. Keep strict FP32 BMM; no precision relaxation / TF32.
        if compile:
            self.score=torch.compile(score_cached,fullgraph=True,dynamic=False)
            self.mask=torch.compile(mask_only,fullgraph=True,dynamic=False)
            self.finish=torch.compile(finish,fullgraph=True,dynamic=False)
            self.candidates=torch.compile(candidate_sides,fullgraph=True,dynamic=False)
        else:
            self.score=score_cached;self.mask=mask_only;self.finish=finish
            self.candidates=candidate_sides
        if fused:
            from .fast_features import fused_finish
            self.finish=fused_finish

    def __call__(self,layer,q,sketch,prepared,step,router):
        score,value=self.score(q,*prepared)
        mask=self.mask(score,router.beta)
        if router.kind=='U-Sol':return mask
        nq=q.shape[1]*self.query_stride
        if q.shape[1]<self.batch:nq-=(-sketch.current_tokens)%self.query_stride
        rem,add=self.candidates(score,mask,sketch.origins,router.beta,128,self.query_stride,nq)
        step=step if isinstance(step,torch.Tensor) else q.new_tensor(float(step),dtype=torch.float32)
        return self.finish(q,sketch,score,mask,rem,add,router.beta,step,value,
                           router.heads[f'{layer}_{router.kind}'])
