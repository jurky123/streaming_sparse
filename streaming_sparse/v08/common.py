"""Immutable experiment identity, explicit output boundaries and paired RNG."""
import hashlib
import json
import os
import random
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def protocol():
    return json.loads((HERE/'configs/protocol.json').read_text())


def source_identity():
    h = hashlib.sha256()
    paths = list(HERE.glob('*.py')) + [HERE/'configs/protocol.json']
    paths += [ROOT/x for x in ('utils/wan_wrapper.py','utils/scheduler.py',
              'wan/modules/causal_model.py','wan/modules/model.py','wan/modules/attention.py',
              'streaming_sparse/v05/primitives.py','streaming_sparse/capture/capture_runner.py',
              'configs/default_config.yaml','configs/self_forcing_dmd.yaml',
              'wan_models/Wan2.1-T2V-1.3B/config.json')]
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def output(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE):
        raise ValueError('v0.8 outputs must remain inside v08')
    p.mkdir(exist_ok=True, parents=True)
    return p


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


def tensor_digest(items):
    h = hashlib.sha256()
    for name, t in sorted(items):
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def paired_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Every external asset is local; respect the user's direct-network preference.
    for key in list(os.environ):
        if key.lower() in ('http_proxy','https_proxy','all_proxy'):
            os.environ.pop(key, None)
    os.environ['HF_HUB_OFFLINE'] = '1'


def require_audit(real=True):
    path = HERE/'reports/phase0_real.json' if real else HERE/'reports/phase0_unit.json'
    if not path.exists(): raise RuntimeError(f'Missing required audit: {path}')
    x = json.loads(path.read_text())
    if not x.get('passed') or x['source_sha256'] != source_identity():
        raise RuntimeError('Audit failed or experiment source changed; rerun Phase0')
    return x


def nmse(prediction, target):
    # One complete external input is one observation, independent of token count.
    return (prediction.float()-target.float()).square().sum()/(target.float().square().sum()+1e-8)


def pair_index(layer, head):
    p = protocol()
    if layer not in p['layers'] or head not in range(p['heads']): raise ValueError('Unknown pair')
    return p['layers'].index(layer)*p['heads']+head
