"""Fixed full-coverage protocol and strict artifact identities."""
import json
from pathlib import Path
from ..learned_cache.io import atomic, digest
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/full_coverage_v1'
E24 = ROOT / 'artifacts/e24_joint_representatives_v1'
BASE = ROOT / 'artifacts/four_gaps_20261031/shared/epoch1.pt'
PRE = ROOT / 'artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'
LAYERS = tuple(range(1, 30))
HEADS = tuple(range(12))
NAMES = ('keys', 'mean', 'detail', 'stats')

def protocol():
    old = json.loads((E24 / 'protocol.json').read_text())
    value = dict(version=1, cases=[c for c in old['cases'] if c['split'] != 'test'],
                 layers=list(LAYERS), heads=list(HEADS), chunks=[2, 5, 8], steps=[0, 2],
                 tiles=old['tiles'], history_pages=338, page_tokens=64, query_tokens=64,
                 representatives=4, probes=16, output_basis=16, detail_rank=8,
                 density=.1, swaps=[16,32], own_protected=True, no_op='strict raw gain > 0',
                 seeds=[17,29], epochs=2, train_events=33408, development_events=8352,
                 events_per_trajectory=4176, test_accessed=False, layer0='Dense',
                 source_checkpoint=digest(PRE), backbone_checkpoint=digest(ROOT/'checkpoints/self_forcing_dmd.pt'),
                 B16=digest(E24/'fit.pt'), basis='reuse frozen E24 train-only B16; no development refit',
                 initialization='original pooling on all 348 positions; E24 A shared Reader warm start',
                 trainable=['pool_k','pool_v','pool_feature','shared E24 Reader'],
                 frozen=['bank','group_v','group_gate','basis','logtheta','backbone'],
                 objective='E24 C: attention NMSE + .1 local response/beta + .1 set loss',
                 schedule='2 full shuffled epochs, original 6 Risk / 2 teacher / 2 random state cycle',
                 teacher='FP64 per-page logZ/output; page_log_center_v3; train/development only',
                 scope='full-coverage C first; old E48 Reader is not claimed retrained on the expanded distribution')
    OUT.mkdir(parents=True, exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('Full-coverage protocol identity changed')
    else: atomic(path,value)
    return value

def case_files(split):
    result=[]
    for case in protocol()['cases']:
        if case['split'] != split: continue
        complete=OUT/'trajectories'/case['id']/'complete.json'
        if not complete.exists(): raise RuntimeError(f'Incomplete capture: {case["id"]}')
        rec=json.loads(complete.read_text())
        if rec['events'] != 4176: raise ValueError(f'Incomplete coverage: {complete}')
        result.extend(OUT/p for p in rec['files'])
    return sorted(result)
