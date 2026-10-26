"""Check the new AV objective against direct original-KV hard attention."""
import argparse
import torch

from .common import config, load_raw, atomic_json, paired_seed, sha
from .core import build, scores, initial_mask, candidates, apply_action
from .labels import GROUPS, feature_path
from .learning import ActionHead, batch, objective, choose


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args()
    root,m=config(a.directory);torch.cuda.set_device(0);paired_seed(312)
    assert m['best_metric']=='nmse'and m['monitor_events']=='all96'
    assert m['training_objective']=='actual_av_nmse'and m['lambda_grid']==[0.]
    # Independent loss/gradient calculation, including invalid-action masking.
    logits=torch.randn(2,3,4,device='cuda',dtype=torch.float64,requires_grad=True)
    valid=torch.ones_like(logits,dtype=torch.bool);valid[0,0,3]=False
    errors=torch.rand_like(logits);b=dict(av_nmse=errors,valid=valid,event_count=2)
    loss,_=objective(logits,b,0.,.4)
    pi=(logits.masked_fill(~valid,-torch.inf)/.4).softmax(-1)
    expected=(pi*errors).sum()/2
    analytic=pi*(errors-(pi*errors).sum(-1,keepdim=True))/.4/2
    gradient=torch.autograd.grad(loss,logits)[0]
    torch.testing.assert_close(loss,expected)
    torch.testing.assert_close(gradient,analytic)
    assert gradient[0,0,3]==0
    layer=m['layers'][0];beta=m['monitor_beta']
    row=next(r for r in m['rows']if r['split']=='train'and r['history_tokens']==4680)
    basis=torch.load(root/f'action_basis/layer{layer:02d}.pt',map_location='cpu',weights_only=True)
    normal=torch.load(root/f'normalization/layer{layer:02d}.pt',map_location='cpu',weights_only=True)
    d=load_raw(root,row,layer)
    params={k:v.cuda()for k,v in basis['parameters'].items()}
    sketch=build(d['kh'],d['vh'],d['kc'],d['vc'],params)
    score,_=scores(d['q'],sketch,content=None)
    mask=initial_mask(score,sketch,beta);actions=candidates(score,mask,sketch.origins,beta,seed=16)
    cached=torch.load(feature_path(root,m['cache'],row,layer,beta),map_location='cpu',weights_only=True)
    keys=torch.cat((d['kh'],d['kc']),1);values=torch.cat((d['vh'],d['vc']),1).double()
    # Match the production float32 QK computation, followed by float64 softmax.
    logits_raw=(torch.bmm(d['q'][:,:2].float(),keys.float().transpose(1,2))/128**.5).double()
    dense=torch.bmm(logits_raw.softmax(-1),values)
    checks={}
    for kind in m.get('groups',GROUPS):
        model=ActionHead(normal[kind]).cuda();b=batch(root,m,[row],layer,beta,kind,basis)
        pred=model(b['x']);loss,_=objective(pred,b,0.,.7)
        pi=(pred.masked_fill(~b['valid'],-torch.inf)/.7).softmax(-1)
        independent=(pi.double()*(cached['base_error'][...,None]-cached['gain']).clamp_min(0).cuda().double()).sum()/cached['energy'].sum().cuda().double()
        torch.testing.assert_close(loss.double(),independent,rtol=2e-6,atol=1e-8)
        choice=choose(pred,b['valid'],b['cost'],0.)
        assert b['valid'].gather(-1,choice[...,None]).all()
        selected=apply_action(mask,actions,choice.flatten())
        token=selected[:,:2].repeat_interleave(sketch.costs.long(),dim=-1)
        raw=torch.bmm(logits_raw.masked_fill(~token,-torch.inf).softmax(-1),values)
        error=(raw-dense).square().sum(-1)
        saved=(cached['base_error']-cached['gain'].gather(-1,choice.cpu()[...,None]).squeeze(-1))[:,:2].cuda()
        torch.testing.assert_close(saved.double(),error,rtol=4e-5,atol=2e-6)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()and p.grad.abs().sum()>0 for p in model.parameters())
        checks[kind]=dict(loss=float(loss),direct_hard_SSE_max_abs=float((saved-error).abs().max()),
                         all_parameter_gradients_finite_nonzero=True,invalid_actions_blocked=True)
    atomic_json(root/'P3_preflight.json',dict(passed=True,groups=checks,layer=layer,
        actual_av_loss_independently_checked=True,gradient_independently_checked=True,
        cached_action_errors_match_direct_original_KV=True,no_huber_loss=True,no_mask_ste=True,
        best_metric='actual_hard_attention_nmse',monitor_events=96,
        learning_sha256=sha(__file__.replace('check_attention.py','learning.py'))))
    print(checks,flush=True)


if __name__=='__main__':main()
