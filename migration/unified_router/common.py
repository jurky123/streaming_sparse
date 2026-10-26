"""Run-local protocol, raw-cache identity, and progress files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from streaming_sparse.v08.common import atomic_json, save, source_identity, paired_seed, sha

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'streaming_sparse/.venv/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def config(directory):
    directory = Path(directory).resolve()
    value = read(directory / 'experiment.json')
    assert value['backbone_source_sha256'] == source_identity()
    return directory, value


def event_key(row):
    return hashlib.sha256(row['sample_id'].encode()).hexdigest()[:24]


def history_key(row):
    return hashlib.sha256(f"{row['trajectory_id']}:{row['history_tokens']}".encode()).hexdigest()[:24]


def update(path, **values):
    atomic_json(path, dict(updated_at=datetime.now(timezone.utc).isoformat(), **values))


def load_raw(directory, row, layer, device='cuda'):
    _, m = config(directory)
    path = Path(m['cache']) / 'events' / event_key(row) / f'layer{layer:02d}.pt'
    import torch
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert value['cache_format'] == 3 and value['sample_id'] == row['sample_id']
    assert value['query_indices'] == row['query_indices']
    assert value['teacher_sha256'] == m['teacher_sha256']
    history = torch.load(value['history_file'], map_location='cpu', weights_only=True, mmap=True)
    assert history['trajectory_id'] == row['trajectory_id']
    assert history['history_tokens'] == row['history_tokens'] and history['layer'] == layer
    return {k: v.to(device) for k, v in dict(q=value['q'], kc=value['kc'], vc=value['vc'],
        kh=history['kh'], vh=history['vh']).items()}
