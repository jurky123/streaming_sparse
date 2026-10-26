import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from .common import atomic_json
from .fast_route import FastRoute,prepare
from .fast_kernels import sparse_attention
from .learning import ActionHead
from .fast_audit import timed

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch._dynamo.config.cache_size_limit=128;torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(a.input,map_location='cuda',weights_only=False)
    q,k,v,mask=(data[t]for t in ['q','k','v','mask']);sketch=data['sketch'];prepared=data['prepared']
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval()
    fast=FastRoute();router=SimpleNamespace(kind='K32-Set',beta=data['beta'],heads={'24_K32-Set':head})
    results={}
    for n in [128,256,512,1024]:
        qt=q[:,:n];score,content=fast.score(qt,*prepared);s=fast.mask(score,router.beta)
        rem,add=fast.candidates(score,s,sketch.origins,router.beta);step=q.new_tensor(3.,dtype=torch.float32)
        fs=lambda:fast.finish(qt,sketch,score,s,rem,add,router.beta,step,content,head)
        fs()
        funcs=dict(score=lambda:fast.score(qt,*prepared),threshold=lambda:fast.mask(score,router.beta),
            candidates=lambda:fast.candidates(score,s,sketch.origins,router.beta),features_MLP_action=fs,
            full_route=lambda:fast(24,qt,sketch,prepared,step,router))
        values={name:timed(fn,20)for name,fn in funcs.items()};results[str(n)]=values;print(n,values,flush=True)
        atomic_json(a.output,results)
    from .fast_flex import flex_attention
    for group in [32,64,128]:
        print('flex',group,flush=True)
        try:
            y=flex_attention(q,k,v,mask,group=group)
            ref=sparse_attention(q,k,v,mask,16)
            diff=y.float()-ref.float()
            results[f'flex{group}']=dict(ms=timed(lambda:flex_attention(q,k,v,mask,group),15),
                nmse=float(diff.square().sum()/ref.float().square().sum()),max_abs=float(diff.abs().max()))
            print(results[f'flex{group}'],flush=True)
        except Exception as e:
            results[f'flex{group}']=dict(error=str(e));print(e,flush=True)
        atomic_json(a.output,results)

if __name__=='__main__':main()
