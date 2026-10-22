"""Immutable history snapshots plus query events, split strictly by prompt ID."""
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def tensor_digest(*tensors):
    digest = hashlib.sha256()
    for x in tensors:
        cpu = x.detach().cpu().contiguous()
        digest.update(str((tuple(cpu.shape), str(cpu.dtype))).encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CaptureWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / 'events.jsonl'

    def write(self, history, event, metadata):
        # Content address ensures cache reuse is based on actual immutable KV,
        # including clean-context updates and eviction, rather than step numbers.
        version = tensor_digest(history['k_hist'], history['v_hist'], history['token_positions'])
        scope = hashlib.sha256(str((metadata['prompt_id'], metadata['layer_id'], metadata['head_id'])).encode()).hexdigest()[:16]
        version = scope + '-' + version
        snapshot = self.root / 'history' / f'{version}.pt'
        if not snapshot.exists():
            atomic_save({**history, 'cache_version': version}, snapshot)
        event_digest = tensor_digest(event['q'], event['k_current'], event['v_current'])[:20]
        name = f"{version}-{metadata['chunk_id']}-{metadata['step_id']}-{event_digest}"
        event_path = self.root / 'queries' / f'{name}.pt'
        if event_path.exists():
            return
        record = {**metadata, 'cache_version': version, 'history': str(snapshot.relative_to(self.root)),
                  'event': str(event_path.relative_to(self.root))}
        atomic_save({**event, **metadata, 'cache_version': version}, event_path)
        with self.manifest.open('a') as f:
            f.write(json.dumps(record) + '\n')


class AttentionDataset(Dataset):
    def __init__(self, root, split=None, splits_path=None):
        self.root = Path(root)
        self.records = [json.loads(line) for line in (self.root / 'events.jsonl').read_text().splitlines()]
        if split:
            splits = json.loads(Path(splits_path or self.root / 'splits.json').read_text())
            sets = [set(splits[x]) for x in ('train', 'val', 'test')]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
                raise ValueError('Prompt leakage between splits')
            allowed = set(splits[split])
            self.records = [r for r in self.records if r['prompt_id'] in allowed]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        history = torch.load(self.root / record['history'], weights_only=True, map_location='cpu')
        event = torch.load(self.root / record['event'], weights_only=True, map_location='cpu')
        if history['cache_version'] != event['cache_version']:
            raise ValueError('Mismatched cache version')
        return {**history, **event}


def query_slices(sample, tile):
    """Retile only contiguous captured Q, never concatenate separated windows."""
    indices = sample['query_indices'].cpu()
    cuts = [0] + (torch.where(indices[1:] != indices[:-1] + 1)[0] + 1).tolist() + [len(indices)]
    if max(b-a for a,b in zip(cuts,cuts[1:])) < tile and sample.get('query_tile',tile) < tile:
        raise ValueError(f'Need contiguous capture windows >= {tile} Q tokens; capture with --query-tile 32 for 8/16/32 comparisons')
    return [slice(start, min(start+tile, end)) for begin,end in zip(cuts,cuts[1:])
            for start in range(begin,end,tile)]


def make_splits(root, train=64, val=16, test=16, seed=0):
    import random
    dataset = AttentionDataset(root)
    prompts = sorted({r['prompt_id'] for r in dataset.records})
    if len(prompts) < train + val + test:
        raise ValueError(f'Requested {train+val+test} prompts, captured {len(prompts)}')
    random.Random(seed).shuffle(prompts)
    splits = dict(train=prompts[:train], val=prompts[train:train + val],
                  test=prompts[train + val:train + val + test])
    (Path(root) / 'splits.json').write_text(json.dumps(splits, indent=2))
    return splits


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--train', type=int, default=64)
    p.add_argument('--val', type=int, default=16)
    p.add_argument('--test', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    args = vars(p.parse_args())
    print(make_splits(**args))
