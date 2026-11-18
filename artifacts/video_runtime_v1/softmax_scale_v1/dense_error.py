"""Compare paired original and candidate latents to the same native Dense reference."""
import argparse
import json
from pathlib import Path

import torch


def errors(x, ref):
    delta=x.double()-ref.double()
    return dict(nmse=float(delta.square().sum()/ref.double().square().sum()),
                mse=float(delta.square().mean()),mae=float(delta.abs().mean()),
                max_abs=float(delta.abs().max()))


def main():
    parser=argparse.ArgumentParser()
    for arg in ('pair','baseline','dense','output'):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    old,new=(torch.load(args.pair,weights_only=True)[i] for i in (0,1))
    dense=torch.load(args.dense,weights_only=True)
    baseline=torch.load(args.baseline,weights_only=True)
    info=[json.loads(p.with_name('result.json').read_text()) for p in (args.baseline,args.dense)]
    assert info[0]['video']==info[1]['video']=='city_walk_s701'
    assert info[1]['mode']=='dense_native' and info[0]['frames']==info[1]['frames']==105
    assert old.shape==new.shape==dense.shape and old.dtype==new.dtype==dense.dtype
    assert torch.equal(old.view(torch.uint8),baseline.view(torch.uint8))
    def compare(x,y,ref):
        a,b=errors(x,ref),errors(y,ref)
        return dict(original=a,candidate=b,difference={k:b[k]-a[k] for k in a})
    result=dict(video=info[1]['video'],shape=list(old.shape),dtype=str(old.dtype),baseline_matches_deployment=True,
                video_error=compare(old,new,dense),
                chunks=[compare(x,y,r) for x,y,r in zip(old.chunk(9,1),new.chunk(9,1),dense.chunk(9,1))])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['video_error'])
    print([(round(r['original']['nmse'],5),round(r['candidate']['nmse'],5)) for r in result['chunks']])


if __name__=='__main__':main()
