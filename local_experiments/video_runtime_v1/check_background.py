"""Background kernel: original numeric fixtures and one city_walk speed case."""
import json
from pathlib import Path
import torch
from . import validate
from .scores_action import fused_background_scores,fused_action_scores
from .profile_route import inputs,capture
from .pipeline_graph import BatchGraphs
from .runtime import risk_from_logits
from ..review_v1.ops import Config
from .benchmark import measure

@torch.no_grad()
def run(factored=False):
    candidate=fused_background_scores
    if factored:
        from .scores_action import fused_factored_scores
        candidate=fused_factored_scores
    validate.grouped_scores=candidate
    validate.run('artifacts/video_runtime_v1/hotspot_round5/'+('factored_validation' if factored else 'background_validation'))
    old,e,identity=inputs(True);rt=BatchGraphs(risk_from_logits,old.phi,Config())
    args=[]
    def save(*x):args.extend(x);return fused_action_scores(*x)
    rt.score_function=save;rt.compute(e,True)
    go,x=capture(lambda:fused_action_scores(*args));gn,y=capture(lambda:candidate(*args))
    out=dict(before=measure(go.replay,25),after=measure(gn.replay,25),max_abs=float((x-y).abs().max()),argmax_mismatches=int((x.flatten(1).argmax(1)!=y.flatten(1).argmax(1)).sum()),sign_mismatches=int(((x.max(-1).values.max(-1).values>0)!=(y.max(-1).values.max(-1).values>0)).sum()),identity=identity)
    Path('artifacts/video_runtime_v1/hotspot_round5/'+('factored.json' if factored else 'background.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--factored',action='store_true');run(p.parse_args().factored)
