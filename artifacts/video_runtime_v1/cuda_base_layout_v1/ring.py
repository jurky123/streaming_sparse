"""Check every native and packed slot across repeated clean commits and reset."""
import json
from pathlib import Path
import torch
from local_experiments.review_v1.runtime import PageStore
from local_experiments.review_v1.ops import Config
from local_experiments.video_runtime_v1.fixed15_attention import PackedAttention
from local_experiments.video_runtime_v1.fixed15_layout import pack_range

def store():
    kv={name:torch.zeros((1,432*64,12,128),device='cuda',dtype=torch.bfloat16) for name in ('k','v')}
    result=PageStore(kv,Config(history_pages=360));result.sink_pages=24;result.contiguous_commit=True
    for name in ('keys','mean','detail','stats'):getattr(result.summary,name).zero_()
    return result

a,b=store(),store()
attn=PackedAttention()
a.packed_writer=attn.commit
for buf in attn.cache(a):buf.zero_()
packed_ref=[torch.zeros_like(buf) for buf in attn.cache(a)]
g=torch.Generator(device='cuda').manual_seed(701)
for n in range(20):
    if n==12:
        a.reset();b.reset();attn.reset()
    k,v=[torch.randn((12,72,64,128),device='cuda',dtype=torch.bfloat16,generator=g) for _ in range(2)]
    summary=type(a.summary)(*(torch.randn_like(getattr(a.summary,key)[:,:72]) for key in ('keys','mean','detail','stats')))
    order=torch.arange(4608,device='cuda').expand(12,-1)
    for s in (a,b):s.commit(k,v,summary,order,n*4608)
    assert (a.n,a.write,a.commits)==(b.n,b.write,b.commits)
    assert all(torch.equal(a.cache[key],b.cache[key]) for key in ('k','v'))
    pack_range[(432,12,16)](b.cache['k'],b.cache['v'],*packed_ref,0)
    assert all(torch.equal(x,y) for x,y in zip(attn.cache(a),packed_ref))
    assert torch.equal(a.token_ids,b.token_ids)
    assert all(torch.equal(getattr(a.summary,key),getattr(b.summary,key)) for key in ('keys','mean','detail','stats'))
Path(__file__).with_name('ring.json').write_text(json.dumps(dict(commits=20,reset_after=12,bitwise_all_slots=True),indent=2))
