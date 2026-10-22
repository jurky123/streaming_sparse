"""Frozen data references, deterministic subsets and source identities."""
import hashlib
import json
from pathlib import Path
from ..v04.identity import source_identity as previous_identity
from ..v04.data import SnapshotData, BalancedSchedule
from ..v02.evaluation_journal import atomic_json
from ..capture.dataset_io import atomic_save, query_slices

HERE=Path(__file__).resolve().parent
DEFAULT_DATA=HERE.parent/'v04/data/full'


def protect_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError('v0.5 outputs must stay inside its own version directory')
    return path


def source_identity():
    h=hashlib.sha256(previous_identity().encode())
    for path in sorted(HERE.glob('*.py')):
        h.update(path.name.encode());h.update(path.read_bytes())
    h.update((HERE/'configs/protocol.json').read_bytes())
    return h.hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def versions(data,phase,role='train'):
    """No random Test selection; all subsets are metadata-only and deterministic."""
    if phase=='phase2' and role not in ('monitor',):return sorted(data.groups)
    protocol=json.loads((HERE/'configs/protocol.json').read_text())
    splits=protocol['splits']
    if phase=='phase1':
        prompts=splits[data.split][:2]
        chosen=[]
        for version,rows in sorted(data.groups.items()):
            r=rows[0]
            if (r['prompt_id'] in prompts and r['seed']==0 and r['head_id'] in (0,5,11)
                and (role=='train' or r['history_tokens']==9360)):
                chosen.append(version)
        return chosen
    # Fixed monitor: one balanced metadata cell per layer/head, all8 tiles/4 steps.
    chosen=[]
    for index,(layer,head) in enumerate(( (l,h) for l in (4,14,24) for h in range(12))):
        prompt=splits['val'][index%4];seed=(index//4)%2;history=(4680,9360,14040)[index%3]
        candidates=[v for v,rows in data.groups.items() if
                    (rows[0]['prompt_id'],rows[0]['seed'],rows[0]['layer_id'],rows[0]['head_id'],rows[0]['history_tokens'])
                    ==(prompt,seed,layer,head,history)]
        if len(candidates)!=1:raise ValueError('Monitor coverage mismatch')
        chosen.append(candidates[0])
    return sorted(chosen)
