"""Freeze existing paired router checkpoints without resuming any training."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import torch
from streaming_sparse.v08.common import HERE,source_identity,sha,tensor_digest,atomic_json,protocol
from streaming_sparse.v08.replay import ReplayDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True);assert not(root/'manifest.json').exists();run=HERE/'runs/seed0'
    specs=dict(mass=run/'warmstart_K/initial.pt',K200=run/'warmstart_K/last.pt',V200=run/'warmstart_V/last.pt',
        Kbest=run/'F-K/best.pt',Kpaused=run/'F-K/recovery.pt')
    ds=ReplayDataset('val',require_complete=True);states={};metadata={}
    paused=json.loads((run/'F-K/USER_PAUSE.json').read_text());assert paused['status']=='paused'
    for name,path in specs.items():
        st=torch.load(path,map_location='cpu',weights_only=True);states[name]=st
        assert st['source_sha256']==source_identity()and st['group'].startswith('F-')and not st['pilot']
        assert st['base_digest']==states['mass']['base_digest']
        for n,v in st['lora'].items():
            assert torch.equal(v,states['mass']['lora'][n]),(name,n)
            if n.endswith('lora_B'):assert torch.count_nonzero(v)==0,(name,n,'Nonzero backbone adapter')
        frozen=root/'checkpoints'/f'{name}.pt';frozen.parent.mkdir(exist_ok=True)
        shutil.copy2(path,frozen)
        metadata[name]=dict(path=str(frozen),original_path=str(path),sha256=sha(frozen),information=st['information'],stage=st['stage'],
            updates=st['updates'],seed=st['seed'],router_sha256=tensor_digest(st['router'].items()),
            lora_sha256=tensor_digest(st['lora'].items()),base_digest=st['base_digest'],
            train_fingerprint=st['provenance']['train_fingerprint'],val_fingerprint=st['provenance']['val_fingerprint'])
    assert states['K200']['updates']==states['V200']['updates']==200
    for key in ('initial_router_digest','train_fingerprint','val_fingerprint'):
        assert states['K200']['provenance'][key]==states['V200']['provenance'][key],key
    assert states['K200']['sampler']==states['V200']['sampler'],'Paired warm-start sample schedules differ'
    assert states['Kpaused']['updates']==paused['updates']==1039
    vb=torch.load(run/'warmstart_V/best.pt',map_location='cpu',weights_only=True)
    assert vb['updates']==0
    assert all(torch.equal(v,vb['router'][n])for n,v in states['mass']['router'].items())
    assert all(torch.count_nonzero(v)==0 for n,v in states['mass']['router'].items()if n.startswith('head.networks.')and n.split('.')[-2]=='2')
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256=source_identity(),teacher_sha256=protocol()['teacher_sha256'],
        val_fingerprint=ds.fingerprint(),val_inputs=len(ds),prompt_ids=sorted(ds.registry['splits']['val']),generation_seeds=[0,1],
        keep=.2,layers=[4,14,24],heads=12,latent_chunks=4,history_max_chunks=3,resolution=[480,832],
        checkpoints=metadata,backbone_and_lora_frozen=True,zero_effective_lora=True,test_used=False,
        training_remains_paused=True,training_pause_update=1039,primary_comparison=['K200','V200'],
        additional_current_K_snapshots=['Kbest','Kpaused'],mass_alias='V warm-start validation-selected best is update0 and equals calibrated mass initialization',
        caution='V-aware has only head warm-start; Kbest/Kpaused include additional router training. This is current-checkpoint evaluation, not a matched main-training comparison or quality ceiling.',
        videos=True,worker_count=8)
    atomic_json(root/'manifest.json',manifest);print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
