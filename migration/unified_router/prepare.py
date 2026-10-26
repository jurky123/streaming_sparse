"""Freeze the user-amended Dense-first, current-P16 experiment protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import torch

from streaming_sparse.v08.replay import ReplayDataset
from .common import ROOT, atomic_json, sha, source_identity, paired_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cache = ROOT / 'streaming_sparse/v10/data/current_p16_densefirst_cache'
    cache.mkdir(parents=True, exist_ok=False)
    old = json.loads((ROOT / 'streaming_sparse/v08/experiments/local_attn_29_20261023/experiment.json').read_text())
    rows = []
    fingerprints = {}
    for split in ['train', 'val']:
        ds = ReplayDataset(split, require_complete=True)
        fingerprints[split] = ds.fingerprint()
        first = set()
        for index, row in enumerate(ds.rows):
            if row['prompt_id'] not in old['splits'][split]:
                continue
            candidates = [dict(row)]
            token = (row['trajectory_id'], row['step'])
            if token not in first:
                first.add(token)
                candidates.insert(0, dict(row, chunk=0, history_tokens=0, history_positions=[0, 0],
                    sample_id=f"{row['trajectory_id']}.chunk0.step{row['step']}"))
            for item in candidates:
                seed = int.from_bytes(hashlib.sha256((item['sample_id'] + ':unified_p16').encode()).digest()[:8], 'little') % (2**63-1)
                item.update(replay_index=index, query_indices=torch.randperm(4680,
                    generator=torch.Generator().manual_seed(seed))[:128].tolist())
                rows.append(item)
    rows.sort(key=lambda r: (r['split'], r['prompt_id'], r['generation_seed'], r['history_tokens'], r['step']))
    assert len(rows) == 640 and len({r['sample_id'] for r in rows}) == 640
    pilot = []
    for split, per_history in [('train', 3), ('val', 2)]:
        ids = old['splits'][split]
        for j, history in enumerate([0, 4680, 9360, 14040]):
            for i in range(per_history):
                pilot.append(next(r['sample_id'] for r in rows if r['split'] == split
                    and r['prompt_id'] == ids[i] and r['generation_seed'] == i % 2
                    and r['history_tokens'] == history and r['step'] == (i+j) % 4))
    model = ROOT / 'streaming_sparse/v08/experiments/local_attn_29_20261023/network_check/dense0/quality/checkpoints/Klast.pt'
    m = dict(version='SlimAV-v0.10-U-currentP16-denseFirst', created_at=datetime.now(timezone.utc).isoformat(),
        directory=str(root), cache=str(cache), backbone_source_sha256=source_identity(),
        teacher_sha256=old['teacher_sha256'], reference_v08_checkpoint=str(model),
        reference_v08_checkpoint_sha256=sha(model), rows=rows, splits=old['splits'], replay_fingerprints=fingerprints,
        pilot_sample_ids=pilot, layers=[4, 14, 17, 23, 24, 27], stress_layer=29,
        first_chunk_dense=True, first_chunk_anchor=False,
        first_chunk_followup_policy='Ordinary historical KV under original visibility; no mandatory retention and no forced exclusion.',
        history_prototypes_by_age={'1': 8, '2': 4, '3+': 2}, current_prototypes=16, page_size=64,
        history_page_partition='Each original4680-token chunk separately; no Page crosses age classes.',
        summary_age_invalidation=True,
        key_dim=64, key_dim_status='candidate Train-fitted projection; not previously validated Slim',
        value_dim=32, true_attention_dim=128, heads=12, queries_per_event=128,
        base_score='Raw prototype LSE; no learned LogZ/Mass calibration and no old global residual head.',
        assignment_initialization='Fresh Train K-only principal directions for H-P2/P4/P8 and C-P16; no inherited Router tensors.',
        foundation_updates=400, foundation_learning_rate=3e-4,
        foundation_objective='K-only Page reconstruction +0.01 usage KL +0.1 QK projection distillation; no V, Mass/LogZ targets or Mask STE.',
        beta_grid=[-.5, 0., .5, 1., 1.5], lambda_grid=[0., .1, .3, 1.],
        baseline_modes=['Dense', 'H-Sol', 'C-Sol', 'U-Sol'],
        groups=['K32-Set', 'V8-Set', 'V32-NoSet', 'V32-Set'], training_seeds=[0, 1],
        min_updates=400, max_updates=800, validate_every=50, learning_rate=3e-4,
        warmup_updates=20, weight_decay=.01, batch_events=8,
        action_candidates_per_side=8, boundary_candidates_per_origin=3, hash_exploration_total=2,
        action_policy='One KEEP/ADD/DROP/SWAP. Each side: up to3 H + up to3 C boundary candidates, then up to2 hash exploration, then boundary fill; never exceeds8.',
        stop_rule='Paired group/seed runs start with400 updates; uniformly extend to800 only under the predeclared paired validation rule.',
        first_chunk_training='Dense audit only; no action-head updates on history0 samples.',
        context_write_dense=True, backbone_and_lora_frozen=True,
        new_test_status='Sealed. Not sampled for development or basis fitting.',
        progression=['P0_math_and_raw_capture', 'P1_baseline_cost_quality', 'P2_action_representation',
                     'P3_four_groups_two_seeds', 'P4_at_most_two_network_candidates'],
        document_sha256=sha(ROOT / 'artifacts/SlimAV_v010_统一KV稀疏_详细设计与实验方案.docx'),
        user_amendments=['Generate first chunk Dense', 'Do not force retention of first-chunk historical KV',
                         'Current Page uses16 prototypes', 'Retrain Router for the new structure',
                         'History usesP8/P4/P2 at1/2/3+ chunks of distance'])
    assert shutil.disk_usage(cache).free > 250 * 1024**3
    atomic_json(root / 'experiment.json', m)
    atomic_json(ROOT / 'migration/unified_router_active.json', dict(directory=str(root)))
    atomic_json(root / 'status.json', dict(status='prepared', stage='P0'))
    print(json.dumps(dict(directory=str(root), events=len(rows), pilot_events=len(pilot), layers=m['layers'])))


if __name__ == '__main__':
    main()
