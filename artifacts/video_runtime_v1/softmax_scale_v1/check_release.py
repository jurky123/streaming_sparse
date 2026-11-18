"""Verify the released fused kernel against the accepted prototype and its fallback."""
import json
from pathlib import Path
import torch
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.video_runtime_v1.weights import ROOT
from scale import variant_for

src=Path(_tail.fn.__code__.co_filename).read_text().split('@tr.jit\ndef _merge')[0].replace('from ..','from local_experiments.')
prototype=variant_for(src,'anchor_guard_named')
data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
q,k,v=(data[name].cuda().contiguous() for name in ('q','k','v'))
keys,mean,det=(data['summary'][name].cuda().contiguous() for name in ('keys','mean','detail'))
basis=data['basis'].cuda().bfloat16().contiguous()
rows=[]
for case in (432,360,288,216,'stress'):
    p=432 if case=='stress' else case
    ids=(data['ids'][...,:p//10]%p).cuda().contiguous()
    if case=='stress':
        q.fill_(10); k.zero_(); k[:,64:128].fill_(10)
        ids.copy_(torch.arange(p//10,device='cuda'))
    outputs=[]
    for kernel,anchor in [(prototype,False),(_tail,True),(_tail,False)]:
        out=torch.empty_like(q)
        kernel[(72,12)](q,keys,mean,det,None,None,None,None,p,4608,12,128,HEAD_PAGES=432,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,K=k,V=v,IDS=ids,EXACT_COUNT=p//10,QK_SCALE=128**-.5,ANCHOR_SCALE=anchor,num_warps=4,num_stages=2,enable_fp_fusion=False)
        outputs.append(out)
    accepted,actual,original=outputs
    row=dict(case=case,prototype_equal=torch.equal(accepted,actual),finite=bool(actual.isfinite().all()),baseline_equal=torch.equal(original,actual))
    print(row,flush=True)
    assert row['prototype_equal'] and row['finite'] and (case!='stress' or row['baseline_equal'])
    rows.append(row)
Path(__file__).with_name('release_module.json').write_text(json.dumps(rows,indent=2))
