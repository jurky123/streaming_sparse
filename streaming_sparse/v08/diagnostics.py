"""Detached current-Student mechanism probes; never called by score forward."""
import math
import torch
from .attention import attend
from ..v05.primitives import select


@torch.no_grad()
def mechanisms(router,q,kh,vh,kc,vc,state,route,keep):
    n=len(kh);size=router.sketch.page_size;b=math.ceil(n/size);pad=b*size-n
    k=torch.nn.functional.pad(kh.float(),(0,0,0,pad)).reshape(b,size,-1)
    v=torch.nn.functional.pad(vh.float(),(0,0,0,pad)).reshape(b,size,-1)
    logits=torch.einsum('qd,btd->qbt',q.float(),k)/math.sqrt(q.shape[-1])
    valid=torch.arange(b*size,device=q.device).reshape(b,size)<n
    logits=logits.masked_fill(~valid[None],-torch.inf)
    true_l=logits.logsumexp(-1)
    true_mu=torch.einsum('qbt,btd->qbd',logits.softmax(-1),v)
    current=q.float()@kc.float().T/math.sqrt(q.shape[-1]);lc=current.logsumexp(-1)
    masses=torch.cat((true_l,lc[:,None]),-1).softmax(-1)
    e=route['estimate'];budget=math.floor(keep*n)
    mass_mask=select(e['logz'],state['costs'],budget)
    true_mass_mask=select(true_l,state['costs'],budget)
    dense=attend(q,kh,vh,kc,vc,torch.ones_like(route['hard']))
    learned=attend(q,kh,vh,kc,vc,route['hard'])
    baseline=attend(q,kh,vh,kc,vc,mass_mask)
    mass_true=attend(q,kh,vh,kc,vc,true_mass_mask)
    # Diagnostic-only V summaries for the K model; this state never enters its score path.
    value_state=router.sketch.build(kh,state['pair'],vh)
    value_estimate=router.sketch.estimate(q,value_state,kc,vc,kind='V')
    weighted_value=((value_estimate['mu']-true_mu).square().sum(-1)*masses[:,:-1]).sum(-1)
    target_energy=dense.square().sum()
    scalar=lambda x:float(x)
    result=dict(logz_signed_bias=scalar((e['logz']-true_l).mean()),
        logz_mae=scalar((e['logz']-true_l).abs().mean()),
        current_mass_signed_bias=scalar((e['current_mass']-masses[:,-1]).mean()),
        historical_mass_l1=scalar((e['mass']-masses[:,:-1]).abs().sum(-1).mean()),
        weighted_value_error=scalar(weighted_value.mean()),
        hard_head_nmse=scalar((learned-dense).square().sum()/(target_energy+1e-8)),
        predicted_mass_head_nmse=scalar((baseline-dense).square().sum()/(target_energy+1e-8)),
        true_mass_head_nmse=scalar((mass_true-dense).square().sum()/(target_energy+1e-8)),
        head_vs_mass_mask_changed=scalar((route['hard']!=mass_mask).any(-1).float().mean()),
        feasible_exchange_count=0,exchange_direction_agreement=None)
    # Metadata-fixed first raw query and first feasible equal-cost selected/unselected swap.
    hard=route['hard'][:1];selected=torch.where(hard[0]>.5)[0].tolist()
    omitted=torch.where(hard[0]<.5)[0].tolist();pair=None
    for remove in selected:
        for add in omitted:
            if state['costs'][add]<=state['costs'][remove]:pair=(remove,add);break
        if pair:break
    if pair:
        swapped=hard.clone();swapped[0,pair[0]]=0;swapped[0,pair[1]]=1
        after=attend(q[:1],kh,vh,kc,vc,swapped)
        actual_gain=(learned[:1]-dense[:1]).square().sum()-(after-dense[:1]).square().sum()
        def estimate_error(mask):
            mh=e['mass'][:1]*mask;mc=e['current_mass'][:1]
            weighted=mh[...,None]*e['mu'][:1]
            # Current content is independently recomputed in the matching content space.
            oc=current[:1].softmax(-1)@(kc if router.kind=='K' else vc).float()
            pred=(weighted.sum(1)+mc[:,None]*oc)/(mh.sum(-1)+mc)[:,None].clamp_min(1e-12)
            return (pred-e['dense'][:1]).square().sum()
        predicted_gain=estimate_error(hard)-estimate_error(swapped)
        result.update(feasible_exchange_count=1,exchange_direction_agreement=bool((actual_gain>0)==(predicted_gain>0)),
                      true_exchange_gain=scalar(actual_gain),predicted_exchange_gain=scalar(predicted_gain))
    return result
