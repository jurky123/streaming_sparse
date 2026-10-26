"""Paired action heads with real-AV policy or legacy gain supervision."""
import torch
from torch import nn
import torch.nn.functional as F
from .labels import feature_path


class ActionHead(nn.Module):
    def __init__(self, normalization):
        super().__init__()
        self.w1=nn.Parameter(torch.randn(12,16,24)*.05)
        self.b1=nn.Parameter(torch.zeros(12,16))
        self.w2=nn.Parameter(torch.randn(12,1,16)*.01)
        self.b2=nn.Parameter(torch.zeros(12,1))
        self.register_buffer('mean',normalization['mean'].clone())
        self.register_buffer('std',normalization['std'].clone())

    def forward(self,x):
        shape=x.shape
        x=(x-self.mean[:,None,None])/self.std[:,None,None]
        y=F.gelu(torch.bmm(x.flatten(1,2),self.w1.transpose(1,2))+self.b1[:,None])
        y=(torch.bmm(y,self.w2.transpose(1,2))+self.b2[:,None]).reshape(shape[:-1])
        return torch.cat((torch.zeros_like(y[...,:1]),y[...,1:]),-1)


def batch(root,m,rows,layer,beta,kind,basis):
    xs=[];gains=[];valid=[];costs=[];av=[]
    for row in rows:
        d=torch.load(feature_path(root,m['cache'],row,layer,beta),map_location='cpu',weights_only=True,mmap=True)
        assert d['sample_id']==row['sample_id']and d['split']==row['split']
        xs.append(torch.cat((d['geometry'][kind],d['key']),-1))
        scale=basis['energy_scale'][:,row['step']]*basis['gain_scale'][:,row['step']]
        gains.append(d['gain']/scale[:,None,None])
        valid.append(d['valid']);costs.append((d['delta_H']+d['delta_C'])/64)
        if m.get('training_objective') == 'actual_av_nmse':
            # Each event gets equal weight, as in hard-action validation.
            # These are exact original-128d attention errors for HARD actions,
            # not attention computed with a soft KV mask.
            av.append((d['base_error'][...,None]-d['gain']).clamp_min(0)/d['energy'].sum().clamp_min(1e-8))
    result = dict(x=torch.cat(xs,1).cuda(),gain=torch.cat(gains,1).cuda(),
        valid=torch.cat(valid,1).cuda(),cost=torch.cat(costs,1).cuda(),types=d['types'].cuda())
    if av:
        result.update(av_nmse=torch.cat(av,1).cuda(),event_count=len(rows))
    return result


def objective(pred,b,lam,temperature):
    if 'av_nmse' in b:
        if lam != 0:
            raise ValueError('Actual-AV policy uses beta to control budget; lambda must be zero')
        # Exact expected attention NMSE over discrete hard candidate actions.
        # Labels are detached; gradients flow only through action probabilities.
        logits=pred.masked_fill(~b['valid'],-torch.inf)
        pi=(logits/temperature).softmax(-1)
        loss=(pi*b['av_nmse'].detach()).sum()/b['event_count']
        return loss,loss.detach()
    errors=F.smooth_l1_loss(pred,b['gain'],reduction='none')
    totals=[];available=[]
    for typ in [1,2,3]:
        use=b['valid']&(b['types'][None,None]==typ)
        count=use.sum(-1)
        totals.append((errors*use).sum(-1)/count.clamp_min(1))
        available.append(count>0)
    gain_loss=(torch.stack(totals,-1).sum(-1)/torch.stack(available,-1).sum(-1).clamp_min(1)).mean()
    utility=(pred-lam*b['cost']).masked_fill(~b['valid'],-torch.inf)
    pi=(utility/temperature).softmax(-1)
    decision=-(pi*(b['gain']-lam*b['cost']).detach()).sum(-1).mean()
    return gain_loss+.1*decision,gain_loss


def choose(pred,valid,cost,lam,margin=0.):
    utility=(pred-lam*cost).masked_fill(~valid,-torch.inf)
    utility=torch.cat((torch.zeros_like(utility[...,:1]),utility[...,1:]),-1)
    value,index=utility.max(-1)
    return torch.where(value>margin,index,torch.zeros_like(index))


@torch.no_grad()
def evaluate(model,root,m,rows,layer,beta,kind,basis,lam=.1,margin=0.):
    records=[]
    for row in rows:
        b=batch(root,m,[row],layer,beta,kind,basis)
        pred=model(b['x'])
        _,loss=objective(pred,b,lam,1.)
        decision=choose(pred,b['valid'],b['cost'],lam,margin)
        d=torch.load(feature_path(root,m['cache'],row,layer,beta),map_location='cpu',weights_only=True,mmap=True)
        ix=decision.cpu()[...,None]
        gain=d['gain'].gather(-1,ix).squeeze(-1)
        kept_h=d['base_H']+d['delta_H'].gather(-1,ix).squeeze(-1)
        kept_c=d['base_C']+d['delta_C'].gather(-1,ix).squeeze(-1)
        nmse=float((d['base_error']-gain).sum()/d['energy'].sum().clamp_min(1e-8))
        tokens=float((kept_h+kept_c).sum())
        diagnostic='expected_nmse' if 'av_nmse' in b else 'huber'
        records.append(dict(prompt_id=row['prompt_id'],sample_id=row['sample_id'],nmse=nmse,
            keep=tokens/(decision.numel()*(row['history_tokens']+4680)),
            keep_H=float(kept_h.mean())/row['history_tokens'],keep_C=float(kept_c.mean())/4680,
            **{diagnostic:float(loss)},acceptance=float((decision!=0).float().mean()),
            actual_worsening=float(((gain<0)&(decision.cpu()!=0)).sum()/
                                   (decision.cpu()!=0).sum().clamp_min(1))))
    return dict(nmse=sum(r['nmse']for r in records)/len(records),keep=sum(r['keep']for r in records)/len(records),
        **{diagnostic:sum(r[diagnostic]for r in records)/len(records)},rows=records)
