import json,statistics,time
from pathlib import Path
import torch
from policy import initial,pool
from scan import counts

def global_pool(risk,mask,own):
    p=risk.size(-1)
    movable=mask & (torch.arange(p,device=risk.device)>=24) & (torch.arange(p,device=risk.device)!=own[...,None])
    r=risk.masked_fill(~movable,torch.inf).argsort(stable=True,dim=-1)[...,:16]
    a=risk.masked_fill(mask,-torch.inf).argsort(stable=True,descending=True,dim=-1)[...,:32]
    valid=movable.gather(-1,r)[..., :, None] & ~mask.gather(-1,a)[...,None,:] & ((r[..., :, None]>=p-72)==(a[...,None,:]>=p-72))
    return r,a,valid

def run():
    root=Path(__file__).parent
    ops=[torch.compile(op,fullgraph=True,dynamic=False) for op in (global_pool,pool)]
    rows=[];start=time.monotonic()
    for p in (216,288,360,432):
        for idx in (11,21):
            torch.manual_seed(701); x=torch.randn(12,72,p,device='cuda')
            own=(torch.arange(72,device='cuda')+p-72).expand(12,72)
            m=initial(x,own,counts(p)[idx]); args=(x,m,own)
            expected=[f(*args) for f in (global_pool,pool)]
            for fn,ref in zip(ops,expected):
                for actual,want in zip(fn(*args),ref):assert torch.equal(actual,want)
            samples=[[],[]]
            for k in range(10):
                for j in (k%2,1-k%2):
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(16):ops[j](*args)
                    b.record();b.synchronize();samples[j].append(a.elapsed_time(b)/16)
            rows.append(dict(p=p,index=idx,ms=[statistics.median(v) for v in samples],median_legal_pairs=[int(v[-1].sum((-2,-1)).median()) for v in expected]))
    (root/'pool_benchmark.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,scope='synthetic risk, 12 heads x 72 query tiles, pool only',rows=rows),indent=2))
if __name__=='__main__':run()
