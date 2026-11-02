"""Independent confirmation protocol for tile-aligned anchored aggregation."""
from pathlib import Path
import json
from . import broad_common as old
from .broad_common import Engine,noise,CHUNKS
from .learned_cache.io import atomic,digest,disk_guard
ROOT=Path('artifacts/tile_anchor_20261030')
NEW=[
 ('rowing_lake','Two people row a small wooden boat across a calm lake at sunrise. A side tracking camera shows synchronized strokes, ripples and reflections.'),
 ('dog_agility','A black and white dog runs through an outdoor agility course, jumping a low barrier and weaving between poles. A wide tracking camera follows the dog.'),
 ('pottery_wheel','A potter shapes a wet clay vase on a spinning wheel. A close steady camera shows fingers pressing the clay and water moving across the surface.'),
 ('market_crowd','People walk through a colorful outdoor market carrying bags, while vendors arrange fruit. A camera moves slowly along the crowded aisle.'),
 ('storm_clouds','Dark storm clouds move over golden fields as wind bends the grass. A wide fixed camera captures changing sunlight and distant rain.'),
 ('aquarium_jellyfish','Translucent jellyfish drift and pulse in a dark aquarium illuminated with blue and violet light. The camera slowly moves closer to their flowing tentacles.')]
def cases(split=None):
    out=old.cases('train')+old.cases('val')
    out += [dict(id=f'test_{name}_s{seed}',split='test',name=name,prompt=prompt,seed=seed) for name,prompt in NEW for seed in [404,505]]
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
        methods=['random_query','random_tile','anchor_query','anchor_tile'],
        code={str(p):digest(p) for p in Path('local_experiments').glob('tile_*.py')})
    path=ROOT/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_text())
        for key in value:
            if key!='code':assert previous[key]==value[key]
    else:atomic(path,value)
    return value
