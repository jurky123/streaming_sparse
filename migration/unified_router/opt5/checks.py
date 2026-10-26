"""Real multilayer/age/head checks, including partial query and page tails."""
import argparse
import torch
from ..common import config,load_raw,atomic_json
from ..network import OnlineRouter
from ..fast_route import prepare
from ..opt2.attention import attention
from .network import OPTIONS,OptimizedRoute
from ..opt4.route import CompactRoute
from ..opt4_fused import install


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory)
    torch.set_num_threads(4);torch._dynamo.config.cache_size_limit=128;torch.backends.cuda.matmul.allow_tf32=False
    router=OnlineRouter(root,m).cuda()
    install();routes={'Opt5':OptimizedRoute(**OPTIONS)}
    ref=CompactRoute('streamgram','fp16','fp16',tile=64)
    rows=[]
    for layer in [1,24,29]:
        for hist,n in [(4680,17),(9360,72),(14040,128)]:
            meta=next(r for r in m['rows']if r['split']=='val'and r['history_tokens']==hist and r['step']==3)
            d=load_raw(root,meta,layer);q=d['q'][:,:n];step=q.new_tensor(3.,dtype=torch.float32)
            k=torch.cat((d['kh'],d['kc']),1);v=torch.cat((d['vh'],d['vc']),1)
            for kind in ['K32-Set','V32-Set']:
                router.kind=kind;sketch=router.sketch(layer,d['kh'],d['vh'],d['kc'],d['vc']);prepared=prepare(sketch,kind)
                reference=ref(layer,q,sketch,prepared,step,router);yr=attention(q,k,v,reference)
                yd=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])[0]
                for name,route in routes.items():
                    state=route.begin(prepared,sketch);actual=route.run(layer,q,sketch,state,step,router)
                    assert actual.any(-1).all()
                    y=attention(q,k,v,actual);assert torch.isfinite(y).all()
                    score,raw=route.score(q,state);mask=route.mask(score,router.beta)
                    r,ad=route.candidates(score,mask,sketch.origins,router.beta,128,1,n)
                    st,cv=route.stats(q,sketch,score,mask,r,ad,raw,state)
                    assert torch.isfinite(st[:,:73]).all()and torch.isfinite(cv).all()
                    row=dict(variant=name,layer=layer,history=hist,kind=kind,queries=q.shape[0]*n,
                        changed_queries=int((actual!=reference).any(-1).sum()),
                        attention_delta_sse=float((y.float()-yr.float()).square().sum()),reference_energy=float(yr.float().square().sum()),
                        attention_sse_to_dense=float((y.float()-yd.float()).square().sum()),
                        reference_sse_to_dense=float((yr.float()-yd.float()).square().sum()),dense_energy=float(yd.float().square().sum()),
                        keep=float((actual*sketch.costs).sum()/(q.shape[0]*n*sketch.costs.sum())),
                        reference_keep=float((reference*sketch.costs).sum()/(q.shape[0]*n*sketch.costs.sum())))
                    rows.append(row);print(row,flush=True)
                    atomic_json(root/'optimization5/checks.json',dict(status='running',rows=rows))
    atomic_json(root/'optimization5/checks.json',dict(status='completed',rows=rows,finite_nonempty=True))


if __name__=='__main__':main()
