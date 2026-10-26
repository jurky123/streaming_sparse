"""Numerical diagnostic: match the compiler's fused multiply-add convention."""
import argparse,json
from pathlib import Path
import torch
import triton as tr
from .common import atomic_json
from .fast_route import FastRoute
from .fast_features import fused_finish
from .learning import ActionHead
from .opt2.score import _content
from .opt2.features import split_finish

def fma_score(q,basis,projected,fma=True):
    h,n,_=q.shape;total=sum(lc.shape[1]for k,lc,v in projected)
    score=torch.empty((h,n,total),device=q.device,dtype=torch.float32)
    has_value=projected[0][2]is not None
    content=torch.empty((h,n,total,32),device=q.device,dtype=torch.float32)if has_value else None
    qq=torch.bmm(q.float(),basis);offset=0
    for key,lc,value in projected:
        b,p=lc.shape[1:];logits=torch.bmm(qq,key.transpose(1,2))
        _content[(tr.cdiv(n,32),b,h)](logits,lc,value,score,content,n,b,p,total,offset,32,max(16,p),has_value,
            num_warps=4,enable_fp_fusion=fma)
        offset+=b
    return score,content

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=data['q'];sketch=data['sketch'];prepared=data['prepared'];beta=data['beta']
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval();fast=FastRoute(True,True)
    rows=[];step=q.new_tensor(3.,dtype=torch.float32)
    for start in range(0,q.shape[1],512):
        qt=q[:,start:start+512];score,value=fast.score(qt,*prepared)
        mask=fast.mask(score,beta);rem,add=fast.candidates(score,mask,sketch.origins,beta)
        ref=fused_finish(qt,sketch,score,mask,rem,add,beta,step,value,head)
        for fma in [False,True]:
            s,v=fma_score(qt,*prepared,fma=fma);m=fast.mask(s,beta);r,a=fast.candidates(s,m,sketch.origins,beta)
            got=split_finish(qt,sketch,s,m,r,a,beta,step,v,head)
            row=dict(start=start,fma=fma,score_exact=torch.equal(s,score),value_exact=torch.equal(v,value),score_max=float((s-score).abs().max()),value_max=float((v-value).abs().max()),changed_queries=int((got!=ref).any(-1).sum()))
            rows.append(row);print(json.dumps(row),flush=True)
    atomic_json(root/'optimization2/fma_probe.json',rows)

if __name__=='__main__':main()
