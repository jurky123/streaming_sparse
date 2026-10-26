"""Assemble actual layer identities, then bridge to the existing video evaluator."""
import argparse
import copy
import json
from pathlib import Path
import torch
from streaming_sparse.v08.common import HERE,save,atomic_json,protocol,sha,source_identity
from migration.paired_router.runtime import copy_pair


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve();m=json.loads((root/'experiment.json').read_text())
    common=torch.load(root/'common_initial.pt',map_location='cpu',weights_only=True)['router']
    parent=torch.load(HERE/'runs/seed0/warmstart_K/initial.pt',map_location='cpu',weights_only=True)
    # LoRA remains an exactly zero-effective compatibility wrapper for all29 layers.
    lora={}
    for layer in m['layers']:
        for name,value in parent['lora'].items():
            if 'blocks.4.'in name:lora[name.replace('blocks.4.',f'blocks.{layer}.')]=torch.zeros_like(value)
    for name,value in parent['lora'].items():lora[name]=value.clone()
    assert lora and all(torch.count_nonzero(v)==0 for n,v in lora.items()if n.endswith('lora_B'))
    bridge=root/'network_check';stage=bridge/'dense0';stage.mkdir(parents=True,exist_ok=True);budgets={}
    for kind in ['K','V']:
        for checkpoint in ['initial','best','last']:
            full=copy.deepcopy(common);updates={}
            for layer in m['layers']:
                cell=torch.load(root/f'training/layer{layer:02d}/{kind}/{checkpoint}.pt',map_location='cpu',weights_only=True)
                for h in range(12):copy_pair(full,cell['router'],h,(layer-1)*12+h)
                updates[layer]=cell['updates']
            full['sketch.normalization_locked'].fill_(True)
            st=dict(source_sha256=m['source_sha256'],teacher_sha256=m['teacher_sha256'],router=full,lora=lora,group='F-'+kind,information=kind,updates=max(updates.values()),
                layer_updates=updates,objective=m['loss'],input_distribution=m['distribution'],stage='local_attention',seed=0)
            if checkpoint=='initial':
                if kind=='K':save(stage/'mass.pt',st)
            else:save(stage/f'runs/seed0/F-{kind}/{checkpoint}.pt',st)
            budgets[kind+':'+checkpoint]=updates
    pp=protocol();pp['layers']=m['layers'];atomic_json(stage/'configs/protocol.json',pp)
    atomic_json(bridge/'experiment.json',dict(source_sha256=source_identity(),stages={'dense0':dict(directory=str(stage),layers=m['layers'],protocol_sha256=sha(stage/'configs/protocol.json'))}))
    atomic_json(root/'assembled.json',dict(bridge=str(bridge),updates_by_layer=budgets,source_sha256=source_identity(),
        quality_note='Local Dense-input Attention supervision; actual29-layer sparse replay/closed-loop videos evaluate accumulated distribution shift.'))
    from migration.paired_router.evaluation import prepare
    prepare(bridge,'dense0');print(json.dumps(dict(bridge=str(bridge),ready=True)),flush=True)


if __name__=='__main__':main()
