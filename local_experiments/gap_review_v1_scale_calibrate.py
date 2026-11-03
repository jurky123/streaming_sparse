"""Train-only mean calibration preserving within-head page score ordering."""
import argparse,json,torch
from pathlib import Path
from .learned_cache.io import atomic,digest
ROOT=Path('artifacts/review_v1_20261102')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--expanded',action='store_true');ap.add_argument('--source');args=ap.parse_args()
    torch.set_num_threads(4)
    source=ROOT/('summary_expand/reef_fish/latest.pt' if args.expanded else 'summary_train/pilot/latest.pt')
    if args.source:source=Path(args.source)
    source_sha=digest(source)
    out=ROOT/('route_expanded' if args.expanded else 'route_scale')
    saved=torch.load(source,map_location='cpu',weights_only=False);rows=[]
    layers=range(1,30) if args.expanded else (1,15,29)
    train_cases=('city_walk','reef_fish') if args.expanded else ('city_walk',)
    data=ROOT/('expanded_probe/calibration' if args.expanded else 'probe/summary_pilot')
    for l in layers:
        prior=torch.nn.functional.softplus(saved['model'][f'layers.{l}.logtheta'])
        pred_sum=torch.zeros(12,dtype=torch.float64);label_sum=pred_sum.clone();events=0
        for case in train_cases:
            assert (data/case/'complete.json').exists()
            pattern=f'calibration_chunk*_layer{l}_step*.pt' if args.expanded else f'calibration_chunk*_layer{l}.pt'
            files=list((data/case).glob(pattern));assert len(files)==(12 if args.expanded else 2)
            for p in files:
                e=torch.load(p,map_location='cpu',weights_only=False)
                assert e['summary_identity']['summary_checkpoint']==source_sha
                assert torch.isfinite(e['features']).all() and torch.isfinite(e['labels']).all(),str(p)
                pred=(e['features'].double()*prior.double()[:,None,None]).sum(-1)
                pred_sum+=pred.sum((1,2));label_sum+=e['labels'].double().sum((1,2));events+=1
        scale=(label_sum/pred_sum.clamp_min(1e-30)).clamp_min(1e-30)
        # Only fixed within-head ranking is deployed. Keep score units within
        # FP32 range; retain the full physical scale separately, never clip labels.
        applied=scale.clamp(1e-8,1e8)
        theta=(prior.double()*applied[:,None]).float()
        assert torch.isfinite(theta).all() and (theta>0).all()
        saved['model'][f'layers.{l}.logtheta']=theta+torch.log(-torch.expm1(-theta))
        rows.append(dict(layer=l,scale=scale.tolist(),physical_log_scale=scale.log().tolist(),applied_scale=applied.tolist(),score_units_rescaled=(applied!=scale).tolist(),events=events))
    identity=dict(summary_sha256=source_sha,train_cases=train_cases,mode='one positive scale/head shared across four risk features; fixed-budget page ranking preserved',dynamic_thresholds_accepted=False)
    atomic(out/'model.pt',dict(model=saved['model'],identity=identity),True);atomic(out/'result.json',dict(identity=identity,rows=rows))
    print(json.dumps(dict(output=str(out),layers=len(rows))),flush=True)
if __name__=='__main__':main()
