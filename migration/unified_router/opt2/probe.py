"""Fixed real-block microbenchmark; never tunes on video quality."""
import argparse,json
from pathlib import Path
import torch
from ..common import atomic_json
from ..fast_route import FastRoute
from ..fast_features import fused_finish
from ..learning import ActionHead
from ..fast_audit import timed
from .features import split_finish
from .score import direct_score

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    out=root/'optimization2';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=data['q'][:,:512];sketch=data['sketch'];prepared=data['prepared'];beta=data['beta']
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval()
    fast=FastRoute(True,True);score,value=fast.score(q,*prepared);mask=fast.mask(score,beta)
    rem,add=fast.candidates(score,mask,sketch.origins,beta);step=q.new_tensor(3.,dtype=torch.float32)
    args=(q,sketch,score,mask,rem,add,beta,step,value,head)
    ref=fused_finish(*args)
    results=dict(old_score_ms=timed(lambda:fast.score(q,*prepared),20),old_finish_ms=timed(lambda:fused_finish(*args),20),split=[],score=[])
    print('baseline',results,flush=True)
    for sw in [4,8,16]:
        for aw in [4,8]:
            actual=split_finish(*args,stat_warps=sw,action_warps=aw)
            r=dict(stat_warps=sw,action_warps=aw,ms=timed(lambda:split_finish(*args,stat_warps=sw,action_warps=aw),20),different_queries=int((actual!=ref).any(-1).sum()))
            results['split'].append(r);print(r,flush=True);atomic_json(out/'probe.json',results)
    for tile in [16,32,64]:
        s,v=direct_score(q,*prepared,tile=tile)
        r=dict(tile=tile,ms=timed(lambda:direct_score(q,*prepared,tile=tile),20),score_max=float((s-score).abs().max()),
            value_max=float((v-value).abs().max()),value_nmse=float((v-value).square().sum()/value.square().sum()))
        mm=fast.mask(s,beta);rr,aa=fast.candidates(s,mm,sketch.origins,beta)
        actual=split_finish(q,sketch,s,mm,rr,aa,beta,step,v,head)
        r['different_queries']=int((actual!=ref).any(-1).sum())
        results['score'].append(r);print(r,flush=True);atomic_json(out/'probe.json',results)

if __name__=='__main__':main()
