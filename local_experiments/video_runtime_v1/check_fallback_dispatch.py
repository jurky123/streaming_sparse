"""Check grouped dispatch indices and inactive actions against direct packed scoring."""
import json
from pathlib import Path
import torch
import triton as tr
from .scores_packed import packed_reduce
from .scores_blocked import fallback_block

@torch.no_grad()
def run():
    torch.manual_seed(701);b,m,r,a,d=3,16,3,5,16
    rand=lambda *s:torch.randn(*s,device='cuda',dtype=torch.float64)*.1
    den=rand(b,m,r,a);state=rand(b,m,3,d);cand=rand(b,m,r+a,34);lb=rand(b,m);rx=rand(b,m,r);ro=rand(b,m,r,d);ax=rand(b,m,a);ao=rand(b,m,a,d);maximum=rand(b,m)
    args=(den,state,cand,lb,rx,ro,ax,ao,maximum);ref=packed_reduce(*args);rows=[]
    for mode in ('all','none','mixed'):
        flags=torch.ones(b*r*a,device='cuda',dtype=torch.bool) if mode=='all' else torch.zeros(b*r*a,device='cuda',dtype=torch.bool)
        if mode=='mixed':flags[::7]=True;flags[-1]=True
        out=torch.full_like(ref,123.)
        fallback_block[(tr.cdiv(b*r*a,16),)](*args,out,flags,b*r*a,r,a,m,d,16,num_warps=4,enable_fp_fusion=False)
        expected=torch.where(flags.reshape(b,r,a),ref,123.)
        error=float((out-expected).abs().max());assert torch.equal(out,expected)
        rows.append(dict(mode=mode,max_abs=error,bitwise_equal=True))
    path=Path('artifacts/video_runtime_v1/hotspot_round8/fallback_dispatch.json');path.write_text(json.dumps(rows,indent=2));print(rows)
if __name__=='__main__':run()
