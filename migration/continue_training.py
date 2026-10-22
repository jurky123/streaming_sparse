"""Explicitly branch a migrated F-K snapshot when replay payload identities change.

Original scientific code is not patched. --new-replay archives the old run, resets
optimizer/sampling/validation selection and begins a separately documented 2000
update continuation from the migrated weights. It is not a paired main experiment.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-replay', action='store_true', help='Explicitly start a NEW continuation branch; not exact resume')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import torch
    from streaming_sparse.v08.common import HERE, paired_seed, source_identity, sha
    from streaming_sparse.v08.replay import ReplayDataset, BalancedReplay
    from streaming_sparse.v08.launcher.worker import require_execution_audit
    inventory = json.loads((ROOT/'migration/inventory.json').read_text())
    origin = ROOT/inventory['recovery_path']
    state = torch.load(origin, map_location='cpu', weights_only=True)
    if state['source_sha256'] != source_identity():
        raise ValueError('Scientific source changed; do not bypass this check')
    train = ReplayDataset('train', require_complete=True)
    val = ReplayDataset('val', require_complete=True)
    exact = train.fingerprint() == state['sampler']['fingerprint'] and val.fingerprint() == state['provenance']['val_fingerprint']
    print(json.dumps(dict(replay_identity_matches=exact, original_updates=state['updates'],
        train_inputs=len(train), val_inputs=len(val), new_replay_requested=args.new_replay), indent=2))
    if not exact and not args.new_replay:
        raise SystemExit('Replay differs. Exact resume refused. Inspect the new data and use --new-replay for an explicitly separate continuation.')
    if args.check_only:
        return
    # These must have been regenerated on the target host, not copied as live gates.
    from streaming_sparse.v08.common import require_audit
    require_audit()
    require_execution_audit()
    if args.new_replay:
        directory = HERE/f"runs/seed{state['seed']}/{state['group']}"
        branch = ROOT/'migration/continuations'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        branch.mkdir(parents=True, exist_ok=False)
        paired_seed(state['seed'])
        new = copy.deepcopy(state)
        new['updates'] = 0
        new['best'] = float('inf')
        new['best_step'] = 0
        new['curve'] = []
        new['optimizer']['state'] = {}
        new['sampler'] = BalancedReplay(train, state['seed']).state_dict()
        new['coverage'] = BalancedReplay(train, state['seed']).coverage()
        new['torch_rng'] = torch.get_rng_state()
        new['provenance'] = {**new['provenance'], 'train_fingerprint':train.fingerprint(), 'val_fingerprint':val.fingerprint()}
        path = branch/'continuation_initial.pt'
        torch.save(new, path)
        (branch/'BRANCH.json').write_text(json.dumps(dict(
            kind='new-data weight continuation; NOT exact resume or original paired formal group',
            original_checkpoint=str(origin), original_sha256=sha(origin), original_updates=state['updates'],
            new_updates_start=0, optimizer='reset', sampler='reset', validation_selection='reset',
            train_fingerprint=train.fingerprint(), val_fingerprint=val.fingerprint(),
            original_source_sha256=source_identity()),indent=2)+'\n')
        # No stale best/curve or old process ownership in the new run directory.
        shutil.move(str(directory), str(branch/'origin_run'))
        directory.mkdir(parents=True)
        shutil.copy2(branch/'BRANCH.json',directory/'CONTINUATION_BRANCH.json')
        origin = path
    for key in list(os.environ):
        if key.lower() in ('http_proxy','https_proxy','all_proxy'):
            os.environ.pop(key, None)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    os.chdir(ROOT)
    command=[sys.executable,'-u','-m','streaming_sparse.v08.launcher.worker',
        'streaming_sparse.v08.train','--stage','main','--group',state['group'],
        '--seed',str(state['seed']),'--resume',str(origin)]
    print('Starting documented single-GPU continuation',flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
