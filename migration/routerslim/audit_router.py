"""R1 algebra, complete parameter VJPs, hard masks and K-only invariance."""
import argparse
import copy
import json
import math
import torch
from migration.routerslim import r1
from streaming_sparse.v08.common import atomic_json,paired_seed,source_identity,sha
from streaming_sparse.v08.router import InformationRouter
from streaming_sparse.v08.attention import masks


def algebra():
    torch.manual_seed(17);dtype=torch.float64
    q=torch.randn(7,128,dtype=dtype);var=torch.rand(11,8,128,dtype=dtype)
    valid=torch.rand(11,8)>.3;valid[:,0]=True
    old=(torch.einsum('qd,bpd->qbp',q.square(),var)*valid).sum(-1)/valid.sum(-1)
    new=q.square()@((var*valid[...,None]).sum(1)/valid.sum(-1)[:,None]).T
    w=torch.randn(8,128,dtype=dtype,requires_grad=True);bias=torch.randn(8,dtype=dtype,requires_grad=True)
    values=torch.randn(11,8,128,dtype=dtype,requires_grad=True)
    weights=torch.randn(7,11,8,dtype=dtype).softmax(-1)
    mu=torch.einsum('qbp,bpd->qbd',weights,values);oc=torch.randn(7,128,dtype=dtype)
    mass=torch.randn(7,12,dtype=dtype).softmax(-1);mh=mass[:,:-1];mc=mass[:,-1]
    dense=(mh[...,None]*mu).sum(1)+mc[:,None]*oc
    residual=mh[...,None]*(mu-dense[:,None])
    old_projection=torch.nn.functional.linear(residual,w,bias)
    low_mu=torch.einsum('qbp,bpr->qbr',weights,torch.nn.functional.linear(values,w))
    low_oc=torch.nn.functional.linear(oc,w)
    low_dense=(mh[...,None]*low_mu).sum(1)+mc[:,None]*low_oc
    new_projection=mh[...,None]*(low_mu-low_dense[:,None])+bias
    torch.testing.assert_close(old,new,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(old_projection,new_projection,atol=1e-10,rtol=1e-10)
    g=torch.randn_like(old_projection)
    ga=torch.autograd.grad((old_projection*g).sum(),(w,bias,values),retain_graph=True)
    gb=torch.autograd.grad((new_projection*g).sum(),(w,bias,values))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    return dict(variance_maxabs=float((old-new).abs().max()),projection_maxabs=float((old_projection-new_projection).abs().max()),
        gradient_maxabs=max(float((a-b).abs().max())for a,b in zip(ga,gb)),passed=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    paired_seed(0);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    router=InformationRouter('K').cuda();router.load_state_dict(state['router'])
    report=dict(passed=False,source_sha256=source_identity(),r1_sha256=r1.identity(),checkpoint_sha256=sha(args.checkpoint),algebra=algebra(),cases=[])
    try:
        for pair in (0,11,12,23,24,35):
            for history in (4680,9360,14040):
                q=torch.randn(256,128,device='cuda')*2;k=torch.randn(history,128,device='cuda')
                v=torch.randn_like(k);kc=torch.randn(128,128,device='cuda');vc=torch.randn_like(kc)
                saved=[]
                for optimized in (False,True):
                    if optimized:r1.install()
                    router.zero_grad(set_to_none=True)
                    st=router.build(k,v,pair);out=router.score(q,kc,vc,st,step=2)
                    result=(out['scores'].square().mean()+out['estimate']['logz'].square().mean())
                    result.backward()
                    saved.append(dict(scores=out['scores'].detach().clone(),
                        hard=masks(out['scores'].detach(),st['costs'],.2)[1],
                        gradients={n:p.grad.detach().clone()for n,p in router.named_parameters()if p.grad is not None}))
                    if optimized:
                        s1=router.score(q,kc,vc,st,step=2)['scores']
                        st2=router.build(k,v*13+7,pair)
                        s2=router.score(q,kc,vc*13-7,st2,step=2)['scores']
                        torch.testing.assert_close(s1,s2,rtol=0,atol=0)
                        r1.uninstall()
                a,b=saved
                torch.testing.assert_close(a['hard'],b['hard'],atol=0,rtol=0)
                assert a['gradients'].keys()==b['gradients'].keys()
                for n,g in a['gradients'].items():torch.testing.assert_close(g,b['gradients'][n],atol=5e-5,rtol=2e-4,msg=lambda m:n+' '+m)
                item=dict(pair=pair,history=history,score_maxabs=float((a['scores']-b['scores']).abs().max()),
                    gradient_maxabs=max(float((g-b['gradients'][n]).abs().max())for n,g in a['gradients'].items()),hard_bitwise_equal=True,k_only_v_invariant=True)
                report['cases'].append(item);print(json.dumps(item),flush=True)
        report['passed']=True
    except BaseException as error:report['error']=repr(error);raise
    finally:r1.uninstall();atomic_json(args.output,report)


if __name__=='__main__':main()
