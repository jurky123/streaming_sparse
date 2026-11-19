from hashlib import sha256
import json,sys
from pathlib import Path
from collections import Counter
root=Path(__file__).parent
prior=root.parent/'rebased_budget_ablation_v1'
manifest=json.loads((prior/'manifest.json').read_text())
manifest['source_sha256']={p:h for p,h in manifest['source_sha256'].items() if not p.startswith('/mnt/data')}
for p,h in manifest['source_sha256'].items():
    assert sha256(Path(p).read_bytes()).hexdigest()==h,p
out={}
for split in ('train','development'):
    rows={}
    for folder in sorted((root/'records'/split).iterdir()):
        counts=Counter();n=0
        for f in folder.glob('*_teacher.pt'):
            ck,la,head,step=f.stem.split('_')[:4]
            assert (folder/(f.name.replace('_teacher.pt','_kv.pt'))).is_file()
            counts[(int(la[1:]),int(head[1:]),int(ck[1:]))]+=1;n+=1
        assert n==1392 and set(counts)=={(l,h,c) for l in range(1,30) for h in range(12) for c in (2,6,20,40)} and max(counts.values())==1
        status=json.loads((folder/'status.json').read_text())
        rows[folder.name]=dict(events=n,seconds=status['wall_seconds'],capture_seconds=sum(status['capture_seconds']))
    out[split]=rows
assert set(out['train']).isdisjoint(out['development'])
out['frozen_sources']=manifest['source_sha256']
out['code_sha256']={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
(root/'identity.json').write_text(json.dumps(out,indent=2))
