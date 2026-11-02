"""No model selection on test: summarize preregistered hard grouping results."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import torch
from .response_hard import grouping, METHODS
from .learned_cache.data import records
from .learned_cache.io import atomic, disk_guard

ROOT=Path('artifacts/response_hard_20261030')

def snapshot():
    snap=ROOT/'code_snapshot';snap.mkdir(exist_ok=True)
    identity=json.loads((ROOT/'identity.json').read_text())
    for source in identity['code']:
        relative=Path(source).resolve().relative_to(Path.cwd())
        dst=snap/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==identity['code'][source]
    shutil.copy2(__file__,snap/Path(__file__).name)
    atomic(ROOT/'snapshot_status.json',dict(status='completed',files=len(identity['code'])+1))

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    before=json.loads((ROOT/'resume_before.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    rows=[]
    for p in ROOT.glob('test/*.json'):rows.extend(json.loads(p.read_text())['rows'])
    breakdown={}
    for field in ['scene','layer','head']:
        breakdown[field]={}
        for value in sorted({r[field] for r in rows}):
            breakdown[field][str(value)]={}
            for count in [1,4]:
                for method in METHODS:
                    selected=[r for r in rows if r[field]==value and r['prototypes']==count and r['grouping']==method]
                    breakdown[field][str(value)][f'{method}_p{count}']=statistics.mean(r['nmse'] for r in selected)
    atomic(ROOT/'breakdown.json',breakdown)
    path=records('test')[0];hp=path.with_name(path.name[:-6]+'_history.pt');hist=torch.load(hp,weights_only=True)
    k=hist['k'].cuda().float();v=hist['v'].cuda().float();bank=torch.load(ROOT/'dictionary.pt',weights_only=True)
    q=bank[hist['layer']].cuda();gen=torch.Generator(device='cuda').manual_seed(123)
    random=torch.randn(q.shape,device='cuda',generator=gen);random=random/random.norm(dim=-1,keepdim=True)*q.norm(dim=-1,keepdim=True)
    timing={}
    for method in METHODS:
        query=random if method=='random_response' else q
        for _ in range(3):grouping(k,v,query,method)
        samples=[]
        for _ in range(20):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();grouping(k,v,query,method);b.record();b.synchronize();samples.append(a.elapsed_time(b))
        timing[method]=dict(median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples))
    audit=dict(resume_preserved_completed_shards=True,resumed_shards=len(before),
        validation_events=len(list(ROOT.glob('val/*.json'))),test_events=len(list(ROOT.glob('test/*.json'))),
        dictionary_bytes=sum(t.numel()*t.element_size() for t in bank.values()),
        preprocessing_scope='3 heads, 5 chunks, includes features, hard pages, hard subgroups, both summaries, permutations and invariant checks; no attention kernel',
        timing=timing,storage=disk_guard(ROOT))
    atomic(ROOT/'audit.json',audit)
    summary=json.loads((ROOT/'summary.json').read_text())
    selected={str(count):min(['response','response_v025','response_v1'],key=lambda m:summary['val'][f'{m}_p{count}']['nmse']) for count in [1,4]}
    atomic(ROOT/'validation_selection.json',selected)
    snapshot()
    print(json.dumps(dict(audit=audit,breakdown=breakdown,selected=selected),indent=2))

if __name__=='__main__':main()
