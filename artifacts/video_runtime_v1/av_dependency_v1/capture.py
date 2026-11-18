"""Profile a single normal JIT launch and save the exact launched cubin."""
import argparse
from pathlib import Path

import torch
import triton
from local_experiments.video_runtime_v1.page_prefetch import _tail, page_tail, coeff_tail
from local_experiments.video_runtime_v1.weights import ROOT

ap=argparse.ArgumentParser()
ap.add_argument('mode',choices=('page','coeff','compat'))
mode=ap.parse_args().mode
data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
q,k,v=[data[name].cuda().contiguous() for name in ('q','k','v')]
keys,mean,detail=[data['summary'][name].cuda().contiguous() for name in ('keys','mean','detail')]
ids=data['ids'][...,:43].cuda().contiguous()
basis=data['basis'].cuda().bfloat16().contiguous()
def launch(kernel,out):
    return kernel[(72,12)](q,keys,mean,detail,None,None,None,None,432,4608,12,128,
        BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,HEAD_PAGES=432,K=k,V=v,IDS=ids,
        ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=43,QK_SCALE=128**-.5,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
ref=torch.empty_like(q)
out=torch.empty_like(q)
if mode=='compat':
    assert page_tail is coeff_tail is _tail
    # First launch in this process must exercise the selected binder.
    launch(coeff_tail,out)
    launch(_tail,ref)
else:
    launch(_tail,ref)
    kernel={'page':page_tail,'coeff':coeff_tail}[mode]
    compiled=launch(kernel,out)
    assert '%waitv =' in compiled.asm['ttgir']
    assert ('threadsPerWarp = [8, 4, 1]' in compiled.asm['ttgir'])==(mode=='coeff')
    Path('/tmp/jzk_dep_'+mode+'.cubin').write_bytes(compiled.asm['cubin'])
    for _ in range(50):
        launch(kernel,out)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    launch(kernel,out)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
assert torch.equal(out.view(torch.uint8),ref.view(torch.uint8))
print(triton.__version__, mode, 'P432 output bitwise equal')
