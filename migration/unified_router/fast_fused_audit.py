import argparse,json
from types import SimpleNamespace
import torch
from .common import atomic_json
from .fast_route import FastRoute
from .fast_features import fused_finish
from .learning import ActionHead
from .fast_audit import timed

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch._dynamo.config.cache_size_limit=128;torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(a.input,map_location='cuda',weights_only=False);q=data['q'];sketch=data['sketch'];prepared=data['prepared']
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval();fast=FastRoute()
    beta=data['beta'];step=q.new_tensor(3.,dtype=torch.float32);out=[];records=[]
    for start in range(0,q.shape[1],512):
        qt=q[:,start:start+512];score,value=fast.score(qt,*prepared);mask=fast.mask(score,beta)
        rem,add=fast.candidates(score,mask,sketch.origins,beta)
        args=(qt,sketch,score,mask,rem,add,beta,step,value,head)
        actual=fused_finish(*args);ref=fast.finish(*args)
        row=dict(start=start,different_bits=int((actual!=ref).sum()),different_queries=int((actual!=ref).any(-1).sum()))
        if start==0:
            row.update(compiled_finish_ms=timed(lambda:fast.finish(*args),20),fused_finish_ms=timed(lambda:fused_finish(*args),20))
        print(json.dumps(row),flush=True);records.append(row);out.append(actual)
    result=torch.cat(out,1)
    atomic_json(a.output,dict(rows=records,original_different_bits=int((result!=data['reference_mask']).sum()),
        original_different_queries=int((result!=data['reference_mask']).any(-1).sum())))

if __name__=='__main__':main()
