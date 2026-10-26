"""Declare paired main training, then expand to Dense0 / sparse1:29."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import torch
from streaming_sparse.v08.common import HERE,atomic_json,paired_seed,protocol,sha,source_identity,tensor_digest,save
from streaming_sparse.v08.replay import ReplayDataset,BalancedReplay
from streaming_sparse.v08.router import InformationRouter


def make_initials(stage,train=None):
    from streaming_sparse.v08.replay import BalancedReplay
    train=train or ReplayDataset('train',require_complete=True);stage=Path(stage)
    for kind in ['K','V']:
        path=stage/f'initial_{kind}.pt'
        if path.exists():continue
        st=torch.load(stage/f'runs/seed0/warmstart_{kind}/best.pt',map_location='cpu',weights_only=True)
        router=InformationRouter(kind,pairs=len(st['router']['scale']));router.load_state_dict(st['router'])
        optimizer=torch.optim.AdamW([dict(params=list(router.parameters()),lr=3e-5,initial_lr=3e-5,name='router')],weight_decay=.01)
        initial=copy.deepcopy(st);initial.update(group='F-'+kind,information=kind,stage='main',updates=0,best=float('inf'),best_step=0,curve=[],
            optimizer=optimizer.state_dict(),sampler=BalancedReplay(train,0).state_dict(),coverage=BalancedReplay(train,0).coverage())
        save(path,initial)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve()
    assert root.is_relative_to(HERE);root.mkdir(parents=True,exist_ok=False)
    paired_seed(0);parent=HERE/'runs/seed0/warmstart_K/initial.pt';st=torch.load(parent,map_location='cpu',weights_only=True)
    assert st['source_sha256']==source_identity()and st['stage']=='heads'and st['updates']==0
    assert all(torch.count_nonzero(v)==0 for n,v in st['lora'].items()if n.endswith('lora_B'))
    specs={}
    for name,layers in [('three',[4,14,24]),('dense0',list(range(1,30)))]:
        p0=protocol();p0['layers']=layers;p0['main']['accumulation']=8;p0['main']['updates']=4000
        p0['scope']='New paired frozen-backbone main training; execution/architecture amendment recorded in experiment.json'
        stage=root/name;(stage/'configs').mkdir(parents=True)
        atomic_json(stage/'configs/protocol.json',p0)
        benchmark=stage/'reports/execution/benchmark.json';benchmark.parent.mkdir(parents=True)
        shutil.copy2(HERE/'reports/execution/benchmark.json',benchmark)
        specs[name]=dict(directory=str(stage),layers=layers,protocol_sha256=sha(stage/'configs/protocol.json'),
            min_updates=2000,max_updates=4000,validate_every=200,global_batch=8,world_size_per_group=4,
            plateau_evaluations=3,plateau_relative_improvement=.005)
    heads=root/'three/runs/seed0'
    for kind in ['K','V']:
        value=copy.deepcopy(st);value.update(information=kind,group='F-'+kind)
        path=heads/f'warmstart_{kind}/best.pt';save(path,value)
    train=ReplayDataset('train',require_complete=True);val=ReplayDataset('val',require_complete=True)
    make_initials(root/'three',train)
    data=dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256=source_identity(),
        teacher_sha256=protocol()['teacher_sha256'],seed=0,stages=specs,groups=['F-K','F-V'],
        physical_gpus={'F-K':[0,1,2,3],'F-V':[4,5,6,7]},
        train_fingerprint=train.fingerprint(),val_fingerprint=val.fingerprint(),train_inputs=len(train),val_inputs=len(val),
        common_parent=str(parent),common_parent_sha256=sha(parent),common_router_sha256=tensor_digest(st['router'].items()),
        backbone_frozen=True,lora_frozen=True,all_router_parameters_trainable=True,keep=.2,page_size=64,prototypes=8,
        current_chunk_always_dense=True,history_max_chunks=3,denoising_steps=[0,1,2,3],context_write_dense=True,
        new_layers='New independent Layer x Head pairs receive Train-only normalization and shared K-only LogZ pre-calibration; only original layers 4/14/24 transfer by actual layer/head identity.',
        calibration_prompts=16,calibration_queries=128,calibration_updates_per_event=4,
        convergence='At paired validation boundaries >=2000, stop both only if neither improved its preceding best by >=0.5% in the last3 validations. Common cap4000; hitting cap is not proof of convergence.',
        lr_schedule='Shared native warmup100/cosine to common4000 cap; clipping1; AdamW/router_lr3e-5/weight_decay.01.',
        evaluation='Complete Val x0 and 8 prompts x2 seeds decoded closed-loop videos; Dense and calibrated Mass controls; Test sealed.',
        sequence=['audit_three','train_three_to_paired_stop','quality_three','prepare_and_calibrate_new_pairs','audit_dense0','train_dense0_to_paired_stop','quality_dense0'],
        old_run_preserved=str(HERE/'runs/seed0/F-K'),old_run_pause_update=1039)
    atomic_json(root/'experiment.json',data);atomic_json(root/'status.json',dict(status='prepared',stage='audit_three'))
    pointer=HERE.parents[1]/'migration/paired_router_active.json';atomic_json(pointer,dict(directory=str(root),status='prepared'))
    print(json.dumps(dict(directory=str(root),source_sha256=source_identity(),stages=specs)),flush=True)


if __name__=='__main__':main()
