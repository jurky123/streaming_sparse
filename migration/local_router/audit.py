"""Independent raw-token output/VJP and batched-original Router comparisons."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from streaming_sparse.v08.common import paired_seed,atomic_json,source_identity,nmse
from streaming_sparse.v08.router import InformationRouter
from streaming_sparse.v08.attention import attend
from streaming_sparse.v08.execution import accelerate
from migration.runtime import router_fast
from .core import page_statistics,output,score,loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve();paired_seed(0);accelerate.install();router_fast.install()
    q=torch.randn(12,16,128,device='cuda')*.3;kh=torch.randn(12,129,128,device='cuda')*.3
    vh=torch.randn_like(kh);kc=torch.randn(12,19,128,device='cuda')*.3;vc=torch.randn_like(kc)
    data=page_statistics(q,kh,vh,kc,vc);data.update(q=q,kh=kh,vh=vh);data['target']=output(data,torch.ones_like(data['page_logz']))
    page=torch.rand(12,16,3,device='cuda').requires_grad_();pred=output(data,page)
    ref=torch.stack([attend(q[i],kh[i],vh[i],kc[i],vc[i],page[i])for i in range(12)])
    torch.testing.assert_close(pred,ref,atol=5e-5,rtol=5e-5)
    upstream=torch.randn_like(pred);g=torch.autograd.grad(pred,page,upstream,retain_graph=True)[0];gr=torch.autograd.grad(ref,page,upstream)[0]
    torch.testing.assert_close(g,gr,atol=2e-5,rtol=2e-4)
    checks=dict(raw_output_max_abs=float((pred-ref).abs().max()),raw_gate_gradient_max_abs=float((g-gr).abs().max()))
    for kind in ['K','V']:
        router=InformationRouter(kind,pairs=348).cuda()
        # Nonzero residual head exercises every content/score path.
        with torch.no_grad():
            for head in range(12):router.head.networks[head][-1].weight.normal_(0,.1)
        reference=copy.deepcopy(router);batched=score(router,data,1,2)['scores'];single=[]
        for h in range(12):
            st=reference.build(kh[h],vh[h],h)
            single.append(reference.score(q[h],kc[h],vc[h],st,step=2)['scores'])
        single=torch.stack(single);torch.testing.assert_close(batched,single,atol=5e-5,rtol=5e-5)
        (batched.square().mean()).backward();single.square().mean().backward();maxg=0.
        for (n,p0),(n1,p1)in zip(router.named_parameters(),reference.named_parameters()):
            assert n==n1 and(p0.grad is None)==(p1.grad is None),n
            if p0.grad is not None:
                torch.testing.assert_close(p0.grad,p1.grad,atol=5e-5,rtol=2e-4,msg=n);maxg=max(maxg,float((p0.grad-p1.grad).abs().max()))
        checks[kind]=dict(score_max_abs=float((batched-single).abs().max()),gradient_max_abs=maxg)
        if kind=='K':
            altered=dict(data);altered.pop('vh');altered.pop('page_value');altered.pop('current_value');altered.pop('exact_current_value');altered.pop('target')
            torch.testing.assert_close(score(router,altered,1,2)['scores'],batched,atol=0,rtol=0);checks['K']['V_independent']=True
    atomic_json(root/'audit.json',dict(passed=True,source_sha256=source_identity(),checks=checks,objective='Attention output ONLY; no backbone graph'));print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
