"""Check physical layer/head transfer and the private stopping-loop transform."""
import argparse
import json
from pathlib import Path
import torch
from migration.paired_router import runtime
from streaming_sparse.v08.common import HERE,atomic_json,paired_seed,source_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve();paired_seed(0);cases=[]
    for kind,path in [('K',HERE/'runs/seed0/F-K/best.pt'),('V',HERE/'runs/seed0/warmstart_V/last.pt')]:
        state=torch.load(path,map_location='cpu',weights_only=True);old=runtime.BaseRouter(kind);old.load_state_dict(state['router']);new=runtime.BaseRouter(kind,pairs=348)
        dst=new.state_dict();dst['sketch.cal_scale'].copy_(state['router']['sketch.cal_scale']);dst['sketch.normalization_locked'].copy_(state['router']['sketch.normalization_locked'])
        for layer,head in [(4,0),(14,5),(24,11)]:
            source=[4,14,24].index(layer)*12+head;target=(layer-1)*12+head;runtime.copy_pair(dst,state['router'],source,target)
            q=torch.randn(9,128);k=torch.randn(129,128);v=torch.randn_like(k);kc=torch.randn(19,128);vc=torch.randn_like(kc)
            x=old.score(q,kc,vc,old.build(k,v,source),step=2)['scores'];y=new.score(q,kc,vc,new.build(k,v,target),step=2)['scores']
            torch.testing.assert_close(x,y,atol=0,rtol=0)
            if kind=='K':
                z=new.score(q,kc,vc*7,new.build(k,v*11,target),step=2)['scores'];torch.testing.assert_close(y,z,atol=0,rtol=0)
            cases.append(dict(kind=kind,layer=layer,head=head,source_pair=source,target_pair=target,scores_bitwise_equal=True))
    manifest,spec=runtime.install_training(root,'three','F-K')
    from streaming_sparse.v08 import train
    assert train.HERE==root/'three'and train.protocol()['main']['updates']==4000
    assert '_paired_stop(step, group, directory, curve)'in(root/'three/train_budget.generated.py').read_text()
    assert manifest['stages']['dense0']['layers']==list(range(1,30))
    atomic_json(root/'structural_checks.json',dict(passed=True,source_sha256=source_identity(),runtime_sha256=runtime.identity(),
        cases=cases,new_pairs=312,total_pairs=348,private_training_root=True,paired_stopping_transform_compiled=True))
    print('Physical layer/head transfer, K-only V invariance, 348-pair geometry and private stop-loop checks passed.',flush=True)


if __name__=='__main__':main()
