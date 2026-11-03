"""Preserve old pilot; refit on train only and report held-out label diagnostics."""
import json,torch
from pathlib import Path
from .review_v1.calibration import fit_theta
from .learned_cache.io import atomic,digest
ROOT=Path('artifacts/review_v1_20261102')
def main():
    torch.set_num_threads(4)
    source=ROOT/'summary_train/pilot/latest.pt';out=ROOT/'route_refit'
    saved=torch.load(source,map_location='cpu',weights_only=False);rows=[]
    for l in (1,15,29):
        train=[]
        for c in (2,5):
            e=torch.load(ROOT/f'probe/summary_pilot/city_walk/calibration_chunk{c}_layer{l}.pt',map_location='cpu',weights_only=False)
            assert e['summary_identity']['summary_checkpoint']==digest(source);train.append(e)
        x=torch.cat([e['features'].flatten(1,2) for e in train],1)[:,None]
        y=torch.cat([e['labels'].flatten(1,2) for e in train],1)[:,None]
        theta,fit=fit_theta(x,y)
        assert fit['max_projected_gradient']<1e-5,fit
        saved['model'][f'layers.{l}.logtheta']=theta+torch.log(-torch.expm1(-theta))
        diagnostics={}
        for case in ('city_walk','bird_flight'):
            es=[torch.load(ROOT/f'probe/summary_pilot/{case}/calibration_chunk{c}_layer{l}.pt',map_location='cpu',weights_only=False) for c in (2,5)]
            xx=torch.cat([e['features'].flatten(1,2) for e in es],1);yy=torch.cat([e['labels'].flatten(1,2) for e in es],1)
            pred=(xx*theta[:,None]).sum(-1)
            diagnostics[case]=dict(prediction_to_label=float(pred.sum()/yy.sum()),normalized_mae=float((pred-yy).abs().sum()/yy.sum()),head_ratios=(pred.sum(-1)/yy.sum(-1).clamp_min(1e-12)).tolist())
        rows.append(dict(layer=l,fit=fit,theta=theta.tolist(),diagnostics=diagnostics))
        print(json.dumps(dict(layer=l,loss=fit['final_loss'],kkt=fit['max_projected_gradient'],diagnostics=diagnostics)),flush=True)
    identity=dict(summary_sha256=digest(source),code_sha256=digest('local_experiments/review_v1/calibration.py'),train_case='city_walk',heldout='bird_flight',layers=[1,15,29],step=0,status='optimizer_convergence_checked; routing/video not yet accepted')
    atomic(out/'model.pt',dict(model=saved['model'],identity=identity),True)
    atomic(out/'result.json',dict(identity=identity,rows=rows))
if __name__=='__main__':main()
