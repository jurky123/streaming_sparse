"""Same Reader results: packed versus split endpoint fields."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from . import reader_reduce as module
from .scores import candidate_pool
from ..review_v1.ops import RouteOps,Config
from .benchmark import measure

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    module.AGG32=True;module.WARPS=4;module.SHARD=128;module.GROUP=1
    rt,e,identity=inputs(True,True);out=rt.compute(e,True)
    z=out[1].index_select(2,e['offsets']);q=e['q'].index_select(2,e['offsets']);u=rt.phi.query(q)
    lm=z.double().logsumexp(-1);lm=lm-lm.logsumexp(-1,keepdim=True)
    mask=RouteOps.select(out[3],e['own'],Config(),check_valid=False)['mask'];r,a=candidate_pool(out[3],mask,e['own'])
    fn=torch.compile(lambda:module.compact_scores(u,e['en'],z,lm,mask,r,a),fullgraph=True,options={'triton.cudagraphs':False})
    go,x=capture(fn);module.PACKED=True;gn,y=capture(fn)
    result=dict(before=measure(go.replay,25),after=measure(gn.replay,25),max_abs=float((x-y).abs().max()),actions_equal=bool(torch.equal(x.flatten(1).argmax(-1),y.flatten(1).argmax(-1))),identity=identity)
    assert torch.allclose(x,y,atol=1e-14,rtol=1e-12)
    Path('artifacts/video_runtime_v1/hotspot_round6/packed_scores.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':run()
