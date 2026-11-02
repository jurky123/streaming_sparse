"""Independent confirmation protocol for tile-aligned anchored aggregation."""
from pathlib import Path
import json
from . import broad_common as old
from .broad_common import Engine,noise,CHUNKS
from .learned_cache.io import atomic,digest,disk_guard
ROOT=Path('artifacts/mass_variance_20261030')
NEW=[
 ('basketball_court','Two basketball players dribble and pass on an outdoor court at sunset. A wide tracking camera follows their quick changes of direction.'),
 ('butterfly_garden','An orange butterfly flutters between purple flowers in a sunny garden. A close tracking camera follows its wings and moving stems.'),
 ('glassblower','A glassblower rotates a glowing glass vessel in a workshop. A steady medium shot shows careful hand movements and changing orange reflections.'),
 ('harbor_cranes','Large cranes load containers beside moving cargo ships in a busy harbor. A slow aerial camera reveals workers and rippling water below.'),
 ('lantern_festival','People carry glowing paper lanterns along a riverside path at night. A moving camera passes the crowd as warm reflections shimmer on the water.'),
 ('arctic_fox','A white arctic fox walks across snow and pauses to shake its fur. A side tracking camera follows it under soft blue winter light.')]

def cases(split=None):
    out=old.cases('train')+old.cases('val')
    out += [dict(id=f'test_{name}_s{seed}',split='test',name=name,prompt=prompt,seed=seed) for name,prompt in NEW for seed in [606,707]]
    return [c for c in out if split is None or c['split']==split]
def guard():
    ROOT.mkdir(parents=True,exist_ok=True)
    return disk_guard(ROOT,min_free_gib=50,max_run_gib=4)
def manifest():
    ROOT.mkdir(parents=True,exist_ok=True)
    for name in ['embeddings','trajectories']:(ROOT/name).mkdir(exist_ok=True)
    for c in cases('train')+cases('val'):
        for dest,source in [(ROOT/'trajectories'/c['id'],old.ROOT/'trajectories'/c['id']),
                            (ROOT/'embeddings'/f'{c["name"]}.pt',old.ROOT/'embeddings'/f'{c["name"]}.pt')]:
            if not dest.exists():dest.symlink_to(source.resolve(),target_is_directory=source.is_dir())
    for name in ['dictionary.pt','dictionary_manifest.json']:
        p=ROOT/name
        if not p.exists():p.symlink_to((old.ROOT/name).resolve())
    value=dict(cases=cases(),checkpoint=json.loads((old.ROOT/'manifest.json').read_text())['checkpoint'],
        epochs=6,updates_per_layer=576,train_tiles=8,queries_per_tile=4,eval_queries=32,
        methods=['control','mass_mean','mass_var'],
        code={str(p):digest(p) for p in Path('local_experiments').glob('mass_*.py')})
    path=ROOT/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_text())
        for key in value:
            if key!='code':assert previous[key]==value[key]
    else:atomic(path,value)
    return value
