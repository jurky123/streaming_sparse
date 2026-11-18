"""Compare finite outputs with entire initial compensation tiles masked out."""
from pathlib import Path
import json
import sys
import torch
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.video_runtime_v1.weights import ROOT
from compensate import variant_for

mode=sys.argv[1]
src=Path(_tail.fn.__code__.co_filename).read_text().split('@tr.jit\ndef _merge')[0].replace('from ..','from local_experiments.')
candidate=variant_for(src,mode)
data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
q,k,v=(data[name].cuda().contiguous() for name in ('q','k','v'))
keys,mean,det=(data['summary'][name].cuda().contiguous() for name in ('keys','mean','detail'))
basis=data['basis'].cuda().bfloat16().contiguous()
ids=torch.arange(43,device='cuda',dtype=data['ids'].dtype).expand_as(data['ids'][...,:43]).contiguous()
rows=[]
for case in ('masked_first','large_positive','large_negative'):
    if case!='masked_first':
        q.fill_(10.)
        k.zero_()
        keys.fill_(10. if case=='large_positive' else -10.)
    outputs=[]
    for ker in (_tail,candidate):
        out=torch.empty_like(q)
        ker[(72,12)](q,keys,mean,det,None,None,None,None,432,4608,12,128,HEAD_PAGES=432,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,K=k,V=v,IDS=ids,EXACT_COUNT=43,QK_SCALE=128**-.5,ANCHOR_SCALE=True,COLUMN_MASS=True,num_warps=4,num_stages=2,enable_fp_fusion=False)
        outputs.append(out.float())
    ref,actual=outputs
    assert ref.isfinite().all()
    finite=actual.isfinite().all().item()
    rows.append(dict(case=case,finite=finite,nmse=float((ref-actual).square().sum()/ref.square().sum()) if finite else None,max_abs=float((ref-actual).abs().max()) if finite else None))
Path(__file__).with_name(mode+'_stress.json').write_text(json.dumps(rows,indent=2))
