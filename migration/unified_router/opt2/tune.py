"""Further fixed-input schedule and algebra audit; exact same masks for attention."""
import argparse,json
from pathlib import Path
import torch
from ..common import atomic_json
from ..fast_route import FastRoute
from ..fast_features import fused_finish
from ..learning import ActionHead
from ..fast_audit import timed
from ..fast_kernels import sparse_attention
from .features import split_finish
from .score import direct_score
from .attention import attention

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=data['q'][:,:512];sketch=data['sketch'];prepared=data['prepared'];beta=data['beta']
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval()
    fast=FastRoute(True,True);score,value=fast.score(q,*prepared);mask=fast.mask(score,beta)
    rem,add=fast.candidates(score,mask,sketch.origins,beta);step=q.new_tensor(3.,dtype=torch.float32)
    args=(q,sketch,score,mask,rem,add,beta,step,value,head);ref=fused_finish(*args)
    result=dict(factor=[],attention=[])
    for factor in [False,True]:
        for w in [4,8]:
            actual=split_finish(*args,action_warps=w,factor=factor)
            row=dict(factor=factor,warps=w,ms=timed(lambda:split_finish(*args,action_warps=w,factor=factor),20),
                different_queries=int((actual!=ref).any(-1).sum()))
            print(row,flush=True);result['factor'].append(row)
    q,k,v,mask=(data[x]for x in ['q','k','v','mask']);ref=sparse_attention(q,k,v,mask)
    for group in [16,32,64]:
        for w in [4,8]:
            for stage in [1,2,3]:
                try:
                    output=attention(q,k,v,mask,group,w,stage)
                    diff=output.float()-ref.float()
                    row=dict(group=group,warps=w,stages=stage,ms=timed(lambda:attention(q,k,v,mask,group,w,stage),15),
                        bit_identical=torch.equal(output,ref),nmse=float(diff.square().sum()/ref.float().square().sum()))
                except Exception as e:row=dict(group=group,warps=w,stages=stage,error=str(e))
                print(row,flush=True);result['attention'].append(row);atomic_json(root/'optimization2/tune.json',result)

if __name__=='__main__':main()
