"""Exercise the anchor fallback with page 1 much larger than page 0."""
from pathlib import Path
import json
import torch
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.video_runtime_v1.weights import ROOT
from local_experiments.video_runtime_v1.page_prefetch import page_tail

data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
q=torch.full_like(data['q'],10,device='cuda').contiguous()
k=torch.zeros_like(data['k'],device='cuda').contiguous()
k[:,64:128].fill_(10.)
v=data['v'].cuda().contiguous()
keys,mean,det=(data['summary'][name].cuda().contiguous() for name in ('keys','mean','detail'))
basis=data['basis'].cuda().bfloat16().contiguous()
ids=torch.arange(43,device='cuda',dtype=data['ids'].dtype).expand_as(data['ids'][...,:43]).contiguous()
rows={}
for name,ker in [('baseline',_tail),('scheduled',page_tail)]:
    out=torch.empty_like(q)
    compiled=ker[(72,12)](q,keys,mean,det,None,None,None,None,432,4608,12,128,HEAD_PAGES=432,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,K=k,V=v,IDS=ids,COLUMN_MASS=True,ANCHOR_SCALE=True,EXACT_COUNT=43,QK_SCALE=128**-.5,num_warps=4,num_stages=2,enable_fp_fusion=False)
    if name=='scheduled':assert '%waitv =' in compiled.asm['ttgir']
    if name=='baseline':ref=out
    rows[name]=dict(finite=torch.isfinite(out).all().item(),equal=torch.equal(out,ref))
Path(__file__).with_name('stress.json').write_text(json.dumps(rows,indent=2))
print(rows)
assert all(r["finite"] and r["equal"] for r in rows.values())
