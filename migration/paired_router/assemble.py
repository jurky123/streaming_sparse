"""Transfer only matching physical layer/head pairs, never broadcast old heads."""
import argparse
import copy
import json
from pathlib import Path
import torch
from migration.paired_router import runtime
from migration.paired_router.prepare import make_initials
from streaming_sparse.v08.common import atomic_json,paired_seed,save,sha,source_identity,tensor_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve()
    manifest,spec,_=runtime.configuration(root,'dense0');stage=Path(spec['directory']);paired_seed(0)
    initial=runtime.BaseRouter('K',pairs=348).state_dict();newpairs=set();sources=[]
    for rank in range(8):
        path=stage/f'calibration/rank{rank}.pt';value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['train_fingerprint']==manifest['train_fingerprint']and value['qk_only']
        for pair in value['pairs']:
            assert pair not in newpairs;runtime.copy_pair(initial,value['router'],pair,pair);newpairs.add(pair)
        sources.append(dict(path=str(path),sha256=sha(path),updates_per_pair=value['updates_per_pair']))
    assert newpairs=={(layer-1)*12+h for layer in range(1,30)if layer not in [4,14,24]for h in range(12)}
    original=torch.load(root/'three/runs/seed0/warmstart_K/best.pt',map_location='cpu',weights_only=True)
    initial['sketch.cal_scale'].copy_(original['router']['sketch.cal_scale']);initial['sketch.normalization_locked'].fill_(True)
    for kind in ['K','V','mass']:
        src=original if kind=='mass'else torch.load(root/f'three/runs/seed0/F-{kind}/best.pt',map_location='cpu',weights_only=True)
        target=copy.deepcopy(initial)
        for old_layer_index,layer in enumerate([4,14,24]):
            for head in range(12):runtime.copy_pair(target,src['router'],old_layer_index*12+head,(layer-1)*12+head)
        assert(target['sketch.feature_count']>0).all()
        value=copy.deepcopy(src);value.update(router=target,information='K'if kind=='mass'else kind,group='F-'+('K'if kind=='mass'else kind),stage='heads',updates=0,best_step=0,best=float('inf'),curve=[],architecture_layers=list(range(1,30)))
        lora={}
        for layer in range(1,30):
            for projection in ['q','k','v','o']:
                prefix=f'blocks.{layer}.self_attn.{projection}.'
                # The frozen adapters have zero effective update. Preserve old
                # adapters by actual layer ID; initialize new zero adapters.
                for label,shape in [('lora_A',(8,1536)),('lora_B',(1536,8))]:lora[prefix+label]=src['lora'].get(prefix+label,torch.zeros(shape))
        assert all(torch.count_nonzero(v)==0 for n,v in lora.items()if n.endswith('lora_B'))
        value['lora']=lora
        destination=stage/'mass.pt'if kind=='mass'else stage/f'runs/seed0/warmstart_{kind}/best.pt';save(destination,value)
    make_initials(stage)
    atomic_json(stage/'initialization_manifest.json',dict(source_sha256=source_identity(),sparse_layers=list(range(1,30)),dense_layers=[0],pairs=348,
        transferred_layers=[4,14,24],new_layers=[i for i in range(1,30)if i not in [4,14,24]],new_pair_count=312,
        calibration_sources=sources,new_pairs_shared_between_K_and_V=True,layer_broadcast=False,
        calibration_split='train',backbone_and_lora_frozen=True))
    print(json.dumps(dict(status='assembled',pairs=348,new_pairs=312)),flush=True)


if __name__=='__main__':main()
