"""Measure one warmed deployed fused attention invocation at 432 pages with NCU."""
import torch
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.video_runtime_v1.weights import ROOT

data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
q,k,v=[data[s].cuda().contiguous() for s in ('q','k','v')]
keys,mean,detail=[data['summary'][s].cuda().contiguous() for s in ('keys','mean','detail')]
basis=data['basis'].cuda().bfloat16().contiguous()
ids=data['ids'][...,:43].cuda().contiguous()
out=torch.empty_like(q)
def run():
    _tail[(72,12)](q,keys,mean,detail,None,None,None,None,432,4608,12,128,HEAD_PAGES=432,
                   BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,K=k,V=v,IDS=ids,
                   EXACT_COUNT=43,QK_SCALE=128**-.5,num_warps=4,num_stages=2,enable_fp_fusion=False)
for _ in range(50):run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
