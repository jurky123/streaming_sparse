"""Recompute only nonfinite offline labels on the same deterministic trajectory."""
import argparse,json,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .rollout_suite import generate_chunk
from .review_v1.model import Model
from .review_v1.observed_runtime import ObservedRuntime
from .review_v1.ops import AttentionOps
from .learned_cache.io import atomic,digest
ROOT=Path('artifacts/review_v1_20261102')
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();root=ROOT/'expanded_probe/calibration'/a.case
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;bad={}
    for p in root.glob('calibration*.pt'):
        e=torch.load(p,map_location='cpu',weights_only=False)
        if not torch.isfinite(e['labels']).all():bad[(e['chunk'],e['step'],e['layer'])]=(p,e)
    if not bad:return
    engine=Engine();m=Model('artifacts/four_gaps_20261031/shared/epoch1.pt')
    checkpoint=ROOT/'distill/city_walk/latest.pt';m.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model']);fixed=[]
    def observe(l,q,k,v,s,selection):
        key=(engine.current['chunk'],engine.current['step'],l)
        if key not in bad:return
        path,e=bad[key];assert e['summary_identity']['summary_checkpoint']==digest(checkpoint)
        tile=int(e['tiles'][0]);offset=torch.linspace(0,63,16,device=q.device).round().long();qs=q[:,tile*64:(tile+1)*64].index_select(1,offset)
        teacher,lz=AttentionOps.evaluate(qs,k,v,s,m[l].basis,torch.ones(12,16,k.shape[1],device=q.device,dtype=torch.bool))
        labels=AttentionOps.residual_labels(qs,k,v,s,m[l].basis,teacher,lz).cpu();assert torch.isfinite(labels).all()
        old=e['labels'].double();valid=torch.isfinite(old);torch.testing.assert_close(labels[valid],old[valid],rtol=1e-3,atol=1e-5)
        backup=ROOT/'invalid_label_archive'/a.case/path.name
        backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():path.replace(backup)
        e['labels']=labels;e['label_precision']='FP64 ratio and scaled norm repair';e['repair_ops_sha']=digest('local_experiments/review_v1/ops.py');atomic(path,e,True)
        row=dict(chunk=key[0],step=key[1],layer=l,max_label=float(labels.max()),max_log_label=float(labels.clamp_min(1e-300).log().max()),count_repaired=int((~valid).sum()))
        fixed.append(row);atomic(root/'label_repair.json',dict(rows=fixed));print(json.dumps(row),flush=True)
    rt=ObservedRuntime(engine,m,observer=observe);emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda()
    try:
        affected={x[0] for x in bad}
        for chunk in range(max(affected)+1):
            if chunk in affected:
                clean=generate_chunk(engine,emb,202,chunk)
                torch.testing.assert_close(clean.cpu(),torch.load(root/f'clean{chunk}.pt',weights_only=True),rtol=0,atol=0)
            else:engine.forward(torch.load(root/f'clean{chunk}.pt',map_location='cuda',weights_only=True),emb,chunk,4)
        assert len(fixed)==len(bad)
    finally:rt.close()
if __name__=='__main__':main()
