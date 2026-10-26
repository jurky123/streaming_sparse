"""Real cached Q/K/V equivalence and warm kernel microbenchmarks."""
import argparse,json,time
from pathlib import Path
import torch
from .common import config,load_raw,atomic_json
from .network import OnlineRouter
from .core import scores,initial_mask,candidates
from .fast_kernels import candidate_sides,sparse_attention
from .fast_route import prepare,score_cached,mask_only,FastRoute


def timed(fn,n=10):
    for _ in range(3):fn()
    torch.cuda.synchronize();s=time.perf_counter()
    for _ in range(n):fn()
    torch.cuda.synchronize();return 1000*(time.perf_counter()-s)/n


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    p.add_argument('--compile',action='store_true');a=p.parse_args()
    root,m=config(a.directory);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    router=OnlineRouter(root,m).cuda();fast=FastRoute(a.compile);results=[]
    for hist in [4680,9360,14040]:
        row=next(r for r in m['rows']if r['split']=='val'and r['history_tokens']==hist and r['step']==3)
        d=load_raw(root,row,24);q=d['q'];k=torch.cat((d['kh'],d['kc']),1);v=torch.cat((d['vh'],d['vc']),1)
        for kind in ['U-Sol','K32-Set','V32-Set']:
            router.kind=kind;sketch=router.sketch(24,d['kh'],d['vh'],d['kc'],d['vc']);prepared=prepare(sketch,kind)
            score,content=scores(q,sketch,content=None if kind=='U-Sol' else kind[:3])
            cached,c=score_cached(q,*prepared)
            torch.testing.assert_close(cached,score,atol=0,rtol=0)
            if content is not None:torch.testing.assert_close(content,c,atol=0,rtol=0)
            mask=initial_mask(score,sketch,router.beta)
            assert torch.equal(mask,mask_only(score,router.beta))
            cand=candidates(score,mask,sketch.origins,router.beta,seed=16)
            rem,add=candidate_sides(score,mask,sketch.origins,router.beta)
            assert torch.equal(rem,cand['remove'][:,9:17]);assert torch.equal(add,cand['add'][:,1:9])
            reference=router.route(24,q,sketch,3)
            print('compiling/routing',hist,kind,flush=True)
            got=fast(24,q,sketch,prepared,3,router)
            changed=int((reference!=got).sum());changed_q=int((reference!=got).any(-1).sum())
            expanded=reference.repeat_interleave(sketch.costs.long(),-1)
            truth=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None],attn_mask=expanded[None])[0]
            row_result=dict(history=hist,kind=kind,mask_different_bits=changed,mask_different_queries=changed_q,
                original_route_ms=timed(lambda:router.route(24,q,sketch,3),3),
                fast_route_ms=timed(lambda:fast(24,q,sketch,prepared,3,router)),attention=[])
            for group in [16,32,64]:
                output,counts=sparse_attention(q,k,v,reference,group,True)
                err=(output.float()-truth.float())
                measurement=dict(group=group,max_abs=float(err.abs().max()),
                    nmse=float(err.square().sum()/truth.float().square().sum()),
                    union_fraction=float(counts.float().mean()/reference.shape[-1]),
                    ms=timed(lambda:sparse_attention(q,k,v,reference,group)))
                assert measurement['nmse']<1e-4,measurement
                row_result['attention'].append(measurement)
            results.append(row_result);print(json.dumps(row_result),flush=True)
            atomic_json(out/'micro.json',results)
    atomic_json(out/'status.json',dict(status='completed',compiled=a.compile))

if __name__=='__main__':main()
