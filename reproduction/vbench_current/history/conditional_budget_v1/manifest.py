import hashlib,json
from pathlib import Path
d=Path(__file__).parent; previous=d.parent/'rebased_retrain_v1'
identity=json.loads((previous/'identity.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(1048576):h.update(block)
 return h.hexdigest()
repo=Path('/home/shuchenweng/jzk/streaming_sparse')
assert all(sha(repo/p)==v for p,v in identity['frozen_sources'].items())
artifacts=[p for pattern in ('*.py','*.sh','*.npz','*.json','budget_*/*.pt','compare*.mp4') for p in d.glob(pattern) if p.name!='manifest.json']
(d/'manifest.json').write_text(json.dumps(dict(frozen=identity['frozen_sources'],train_dev=sha(previous/'identity.json'),artifacts={str(p.relative_to(d)):sha(p) for p in sorted(set(artifacts))}),indent=2))
