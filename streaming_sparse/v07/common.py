"""Source/data identities and immutable experiment boundaries."""
from pathlib import Path
import hashlib
import json
from ..v06.common import (SnapshotData, BalancedSchedule, DEFAULT_DATA, atomic_json, atomic_save,
                          query_slices, file_sha, monitor_versions, tensor_digest)
from ..v06.common import source_identity as predecessor_identity
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def protocol():return json.loads((HERE/'configs/protocol.json').read_text())


def source_identity():
    h=hashlib.sha256(predecessor_identity().encode())
    for p in sorted(HERE.glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    h.update((HERE/'configs/protocol.json').read_bytes())
    for p in sorted((HERE/'design').glob('*.md')):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def output(path):
    p=Path(path).resolve()
    if not p.is_relative_to(HERE):raise ValueError('v0.7 outputs must stay in v07')
    p.mkdir(parents=True,exist_ok=True)
    return p


def pair_index(layer,head):
    if layer not in (4,14,24) or head not in range(12):raise ValueError('Unknown pair')
    return (4,14,24).index(layer)*12+head


def fixed_samples(data):
    # Locked metadata only, all36 pairs; no selection based on model error.
    return [dict(snapshot=v,step=i%4,tile=(i//4)%8,query=(i*7)%16)
            for i,v in enumerate(monitor_versions(data))]


def train_monitor_versions(ds):
    p=protocol();chosen=[]
    for i,(layer,head) in enumerate(( (l,h) for l in (4,14,24) for h in range(12))):
        wanted=(p['train_prompts'][i%16],(i//16)%2,layer,head,p['histories'][i%3])
        found=[v for v,rs in ds.groups.items() if tuple(rs[0][k] for k in ('prompt_id','seed','layer_id','head_id','history_tokens'))==wanted]
        if len(found)!=1:raise ValueError('Train monitor coverage mismatch')
        chosen+=found
    return sorted(chosen)
