"""Immutable v0.5 dependencies, v0.6-only outputs and preregistered data views."""
import hashlib
import json
from pathlib import Path
from ..v05.common import (DEFAULT_DATA, SnapshotData, BalancedSchedule, atomic_json,
                          atomic_save, query_slices, file_sha, versions as old_versions)
from ..v05.common import source_identity as previous_identity

HERE = Path(__file__).resolve().parent


def protect_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError('All v0.6 outputs must stay inside streaming_sparse/v06')
    return path


def protocol():
    return json.loads((HERE / 'configs/protocol.json').read_text())


def source_identity():
    h = hashlib.sha256(previous_identity().encode())
    for path in sorted(HERE.glob('*.py')):
        h.update(path.name.encode()); h.update(path.read_bytes())
    h.update((HERE / 'configs/protocol.json').read_bytes())
    return h.hexdigest()


def monitor_versions(data):
    if data.split != 'val':
        raise ValueError('Validation monitor must use Val prompts')
    return old_versions(data, 'phase2', 'monitor')


def tensor_digest(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def fixed_diagnostic_samples(data):
    """Metadata-only, one query per layer/head, before any result is inspected."""
    chosen = monitor_versions(data)
    return [dict(cache_version=v, step_id=i % 4, tile_id=(i // 4) % 8,
                 query_in_tile=(i * 7) % 16) for i, v in enumerate(chosen)]
