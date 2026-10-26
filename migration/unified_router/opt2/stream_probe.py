import argparse,json
from pathlib import Path
import torch
import triton as tr
from ..common import atomic_json
from ..fast_route import FastRoute
from ..fast_features import fused_finish
from ..learning import ActionHead
from ..fast_audit import timed
from .streaming import score_only,streaming_stats
from .candidate_actions import _candidate_actions
from .features import _statistics

def stream_finish(q,sketch,score,mask,rem,add,beta,step,raw,projected,head,gp=16,tile=16):
    stats,cv=streaming_stats(q,sketch,score,mask,rem,add,raw,projected,gp,tile)
    h,n,b=score.shape;out=torch.empty_like(mask)
    _candidate_actions[(h*n,)](stats,score,cv,mask,sketch.costs,sketch.origins,rem,add,
        head.w1,head.b1,head.w2,head.b2,head.mean,head.std,step,out,n,b,tr.next_power_of_2(b),
        sketch.history_tokens,sketch.current_tokens,beta,num_warps=4,enable_fp_fusion=False)
    return out

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=data['q'][:,:512];sketch=data['sketch'];prepared=data['prepared'];beta=data['beta']
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval()
    fast=FastRoute(True,True);score,value=fast.score(q,*prepared);s,raw=score_only(q,*prepared)
    mask=fast.mask(s,beta);rem,add=fast.candidates(s,mask,sketch.origins,beta);step=q.new_tensor(3.,dtype=torch.float32)
    ref=fused_finish(q,sketch,s,mask,rem,add,beta,step,value,head)
    refstats=torch.empty((q.shape[0]*q.shape[1],80),device=q.device,dtype=torch.float32)
    _statistics[(len(refstats),)](s,value,mask,sketch.costs,sketch.origins,refstats,q.shape[1],s.shape[-1],tr.next_power_of_2(s.shape[-1]),num_warps=4,enable_fp_fusion=False)
    result=dict(score_only_ms=timed(lambda:score_only(q,*prepared),20),settings=[])
    for gp in [8,16,32]:
        for tile in [16,32]:
            args=(q,sketch,s,mask,rem,add,beta,step,raw,prepared[1],head,gp,tile)
            actual=stream_finish(*args);stats,cv=streaming_stats(q,sketch,s,mask,rem,add,raw,prepared[1],gp,tile)
            row=dict(gp=gp,tile=tile,ms=timed(lambda:stream_finish(*args),15),
                different_queries=int((actual!=ref).any(-1).sum()),center_max=float((stats[:,:64]-refstats[:,:64]).abs().max()),
                variance_max=float((stats[:,64:66]-refstats[:,64:66]).abs().max()))
            result['settings'].append(row);print(row,flush=True);atomic_json(root/'optimization2/stream_probe.json',result)

if __name__=='__main__':main()
