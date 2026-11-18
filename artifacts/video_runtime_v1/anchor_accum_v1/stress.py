"""Compare online normalization, an unguarded anchor and its fallback under large logits."""
from pathlib import Path
import json
import torch
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.video_runtime_v1.weights import ROOT
from accum import variant_for

data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
src=Path(_tail.fn.__code__.co_filename).read_text().split('@tr.jit\ndef _merge')[0].replace('from ..','from local_experiments.')
q=torch.full_like(data['q'],10,device='cuda').contiguous()
k=torch.zeros_like(data['k'],device='cuda').contiguous()
k[:,64:128].fill_(10.)
v=data['v'].cuda().contiguous()
keys,mean,det=(data['summary'][name].cuda().contiguous() for name in ('keys','mean','detail'))
basis=data['basis'].cuda().bfloat16().contiguous()
ids=torch.arange(43,device='cuda',dtype=data['ids'].dtype).expand_as(data['ids'][...,:43]).contiguous()
rows={}
for name,ker in [('baseline',_tail),('mass64',variant_for(src,'mass64')),('release',_tail)]:
    out=torch.empty_like(q)
    ker[(72,12)](q,keys,mean,det,None,None,None,None,432,4608,12,128,HEAD_PAGES=432,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,K=k,V=v,IDS=ids,COLUMN_MASS=name=='release',ANCHOR_SCALE=True,EXACT_COUNT=43,QK_SCALE=128**-.5,num_warps=4,num_stages=2,enable_fp_fusion=False)
    if name=='baseline':ref=out
    rows[name]=dict(finite=torch.isfinite(out).all().item(),equal=torch.equal(out,ref))
Path(__file__).with_name('stress.json').write_text(json.dumps(rows,indent=2))
print(rows)
assert all(r["finite"] and r["equal"] for r in rows.values())
