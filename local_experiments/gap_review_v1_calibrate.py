"""Pilot calibration: train scenes only, fixed10 budget; no dynamic claim."""
import json
from pathlib import Path
import torch
from .learned_cache.io import atomic,digest
from .review_v1.calibration import fit_theta
ROOT=Path('artifacts/review_v1_20261102')
def main():
    source=ROOT/'summary_train/pilot/latest.pt';data=ROOT/'probe/summary_pilot/city_walk';out=ROOT/'route_pilot'
    assert (data/'complete.json').exists()
    saved=torch.load(source,map_location='cpu',weights_only=False);rows=[]
    for l in (1,15,29):
        events=[torch.load(data/f'calibration_chunk{c}_layer{l}.pt',map_location='cpu',weights_only=False) for c in (2,5)]
        assert all(e['summary_identity']['summary_checkpoint']==digest(source) for e in events)
        f=torch.cat([e['features'].flatten(1,2) for e in events],1)[:,None]
        labels=torch.cat([e['labels'].flatten(1,2) for e in events],1)[:,None]
        theta,fit=fit_theta(f,labels)
        saved['model'][f'layers.{l}.logtheta']=theta.expm1().log()
        rows.append(dict(layer=l,theta=theta.tolist(),fit=fit))
    identity=dict(summary_sha256=digest(source),train_case='city_walk',source_events=[2,5],step=0,layers=[1,15,29],status='pilot_only; no downstream training, full-layer coverage or dynamic-threshold acceptance')
    atomic(out/'model.pt',dict(model=saved['model'],identity=identity),True)
    atomic(out/'result.json',dict(identity=identity,rows=rows));print(json.dumps(identity),flush=True)
if __name__=='__main__':main()
