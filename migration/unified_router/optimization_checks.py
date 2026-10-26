"""Tail-page/empty-pool tests and multi-layer real-cache numerical audit."""
import argparse
from pathlib import Path
import torch
from .common import config,load_raw,atomic_json
from .network import OnlineRouter
from .core import scores,initial_mask,candidates
from .fast_route import FastRoute,prepare
from .fast_kernels import candidate_sides,sparse_attention

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory)
    torch.set_num_threads(4);torch._dynamo.config.cache_size_limit=128;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(213)
    checks=[]
    for n in [17,72,128]:
        score=torch.zeros(12,n,148,device='cuda');origins=torch.cat((torch.zeros(74),torch.ones(74))).cuda()
        for case in ['all','one','mixed']:
            mask=torch.ones_like(score,dtype=torch.bool)if case=='all'else torch.zeros_like(score,dtype=torch.bool)
            if case=='one':mask[...,147]=True
            if case=='mixed':mask[...,::7]=True
            ref=candidates(score,mask,origins,1.09,seed=16);rem,add=candidate_sides(score,mask,origins,1.09)
            assert torch.equal(rem,ref['remove'][:,9:17]);assert torch.equal(add,ref['add'][:,1:9])
            q=torch.randn((12,n,128),device='cuda',dtype=torch.bfloat16)
            k=torch.randn((12,9360,128),device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
            costs=torch.tensor(([64]*73+[8])*2,device='cuda')
            truth=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None],attn_mask=mask.repeat_interleave(costs,-1)[None])[0]
            actual=sparse_attention(q,k,v,mask);diff=actual.float()-truth.float()
            nmse=float(diff.square().sum()/truth.float().square().sum());assert nmse<1e-4
            checks.append(dict(n=n,case=case,candidates_bit_identical=True,attention_nmse=nmse))
    router=OnlineRouter(root,m).cuda();fast=FastRoute(compile=True,fused=True);rows=[]
    for layer in [1,24,29]:
        for hist in [4680,9360,14040]:
            row=next(r for r in m['rows']if r['split']=='val'and r['history_tokens']==hist and r['step']==3)
            d=load_raw(root,row,layer)
            for kind in ['K32-Set','V32-Set']:
                router.kind=kind;sketch=router.sketch(layer,d['kh'],d['vh'],d['kc'],d['vc']);prepared=prepare(sketch,kind)
                ref=router.route(layer,d['q'],sketch,3);got=fast(layer,d['q'],sketch,prepared,3,router)
                assert got.any(-1).all()
                rows.append(dict(layer=layer,history=hist,kind=kind,queries=got.shape[0]*got.shape[1],
                    changed_queries=int((got!=ref).any(-1).sum()),changed_bits=int((got!=ref).sum())))
    atomic_json(root/'optimization/correctness.json',dict(edge_checks=checks,real_rows=rows,
        total_real_queries=sum(r['queries']for r in rows),total_changed_queries=sum(r['changed_queries']for r in rows),
        finite_and_nonempty=True,tf32_disabled=not torch.backends.cuda.matmul.allow_tf32))
    print('Checks finished:',sum(r['changed_queries']for r in rows),'changed /',sum(r['queries']for r in rows),'real queries',flush=True)

if __name__=='__main__':main()
