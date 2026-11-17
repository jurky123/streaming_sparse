"""Same-input route/tail acceptance before the video test."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .pipeline_graph import BatchGraphs
from .runtime import risk_from_logits
from ..review_v1.ops import Config
from .benchmark import measure

@torch.no_grad()
def run(projection_float=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    old,e,identity=inputs(True)
    new=BatchGraphs(risk_from_logits,old.phi,Config(),compile_route=True,compile_tail=True,action_kernel=True,tensor_math=True,projection_float=projection_float)
    from .attention import compensate
    tail=torch.compile(compensate,fullgraph=True,options={'triton.cudagraphs':False})
    go,x=capture(lambda:old.compute(e,True));gn,y=capture(lambda:new.compute(e,True))
    gt,ox=capture(lambda:tail(x[1],e['s'],e['basis'],x[2],e['haar_float']))
    gu,oy=capture(lambda:new.compensate(y[1],e['s'],e['basis'],y[2],e['haar_float']))
    def err(a,b):return dict(nmse=float((a.double()-b.double()).square().sum()/a.double().square().sum()),max_abs=float((a-b).abs().max()))
    report=dict(route_mask_mismatches=int((x[2]!=y[2]).sum()),risk=err(x[3],y[3]),logits=err(x[1],y[1]),tail=err(ox[0],oy[0]),lse=err(ox[1],oy[1]),before_route=measure(go.replay,25),after_route=measure(gn.replay,25),before_tail=measure(gt.replay,25),after_tail=measure(gu.replay,25),identity=identity)
    assert torch.isfinite(oy[0]).all()
    path=Path('artifacts/video_runtime_v1/hotspot_round5');path.mkdir(parents=True,exist_ok=True);(path/('projection_float.json' if projection_float else 'pipeline.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--projection-float',action='store_true');run(p.parse_args().projection_float)
