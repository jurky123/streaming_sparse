"""Explicit local Attention-only protocol, fixed independent raw query manifest."""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import shutil
import torch
from streaming_sparse.v08.common import HERE,atomic_json,protocol,source_identity,sha,save,paired_seed
from streaming_sparse.v08.replay import ReplayDataset
from streaming_sparse.v08.router import InformationRouter
from migration.paired_router.runtime import copy_pair


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--cache',required=True);a=p.parse_args()
    root=Path(a.directory).resolve();cache=Path(a.cache).resolve();root.mkdir(parents=True,exist_ok=False);cache.mkdir(parents=True,exist_ok=False)
    assert root.is_relative_to(HERE);paired_seed(0);rows=[];splits={};fingerprints={}
    for split,count in [('train',16),('val',4)]:
        ds=ReplayDataset(split,require_complete=True);ids=sorted(ds.registry['splits'][split])[:count];splits[split]=ids;fingerprints[split]=ds.fingerprint()
        for index,row in enumerate(ds.rows):
            if row['prompt_id']not in ids:continue
            seed=int.from_bytes(hashlib.sha256((row['sample_id']+':local_uniform128').encode()).digest()[:8],'little')%(2**63-1)
            ix=torch.randperm(4680,generator=torch.Generator().manual_seed(seed))[:128].tolist()
            rows.append(dict(**row,replay_index=index,query_indices=ix))
    histories=sum(2*(r['history_tokens']+4680)*12*128*2 for r in rows)*29
    stats=sum(12*128*((r['history_tokens']+63)//64)*128*4 for r in rows)*29
    # Deduplicated history + all page-value sufficient statistics, conservatively.
    estimate=histories+stats+30*1024**3;free=shutil.disk_usage(cache).free;assert estimate<free*.8,(estimate,free)
    parent=HERE/'runs/seed0/warmstart_K/initial.pt';old=torch.load(parent,map_location='cpu',weights_only=True)
    router=InformationRouter('K',pairs=348);state=router.state_dict()
    for j,layer in enumerate([4,14,24]):
        for head in range(12):copy_pair(state,old['router'],j*12+head,(layer-1)*12+head)
    router.load_state_dict(state);router.sketch.normalization_locked.fill_(False)
    save(root/'common_initial.pt',dict(router=router.state_dict(),source_sha256=source_identity(),teacher_sha256=protocol()['teacher_sha256']))
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256=source_identity(),teacher_sha256=protocol()['teacher_sha256'],
        directory=str(root),cache=str(cache),rows=rows,splits=splits,replay_fingerprints=fingerprints,layers=list(range(1,30)),
        heads=12,queries_per_event=128,query_policy='Fixed uniform independent queries; no pooling. Same manifest for K/V; both training and validation use all128.',
        loss='Per-layer concatenated12-head Attention-output SSE / target energy; ONLY this loss, weight1; no x0, LogZ, anchor, old-mask or teacher-mass target.',
        target='True original128-d KV attention, represented by exact continuous-page sufficient statistics; FP32 QK, FP64 exp/reductions, FP32 stored page AV means. Audit against direct raw-KV output and mask VJP.',
        distribution='Frozen Dense-backbone QKV; local surrogate only, subsequent actual sparse rollout required.',
        keep=.2,page_size=64,prototypes=8,max_history_chunks=3,current_chunk_always_retained=True,context_write_dense=True,
        backbone_frozen=True,lora_frozen=True,seed=0,initialization='Independent Layer x Head params. Actual layers4/14/24 transfer shared calibrated Mass initial; all other layers independent fresh common K/V initial. Train-only normalization per pair.',
        batch_events=8,training_queries_per_event=128,min_updates=300,max_updates=800,validate_every=50,lr=1e-4,warmup=30,clip_per_layer=1.,
        stopping='After300 updates stop both K/V for each layer together if neither improves its previous best by >=0.5% in last3 validations; otherwise extend up to800. Hitting cap is not proof of convergence.',
        physical_gpus=list(range(8)),parallelism='Independent layer shards on8 GPUs; both K/V on each shard; identical event/query order and initialization.',
        cache_estimated_upper_bytes=estimate,cache_free_bytes_at_start=free,test_used=False,old_network_run='frozen_pair_dense0_20261023 is paused and preserved')
    atomic_json(root/'experiment.json',manifest);atomic_json(root/'status.json',dict(status='prepared',stage='audit'))
    atomic_json(HERE.parents[1]/'migration/local_router_active.json',dict(directory=str(root)))
    print(dict(directory=str(root),events=len(rows),train=sum(r['split']=='train'for r in rows),val=sum(r['split']=='val'for r in rows),cache_upper_TB=estimate/1e12))


if __name__=='__main__':main()
