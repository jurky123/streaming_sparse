"""Fail-closed coverage audit and native Model export for C+Risk deployment."""
import argparse,collections,json,re
import torch
from .common import *
from ..learned_cache.io import RunLock

def audit_data():
    preflight=json.loads((OUT/'joint_preflight.json').read_text())
    assert preflight['passed'] and preflight['positions']==348 and preflight['frozen_exact']
    result={}
    for split,expected in [('train',33408),('development',8352)]:
        paths=case_files(split);assert len(paths)==expected
        counts=collections.Counter()
        for case in protocol()['cases']:
            if case['split']!=split:continue
            rec=json.loads((OUT/'trajectories'/case['id']/'complete.json').read_text())
            assert rec['protocol_sha']==digest(OUT/'protocol.json')
            assert len(rec['coverage'])==348 and set(rec['coverage'].values())=={12}
            seen=set()
            for rel in rec['files']:
                path=OUT/rel
                if not path.is_file():raise FileNotFoundError(path)
                match=re.fullmatch(r'c(\d+)_l(\d+)_s(\d+)_h(\d+)_t(\d+)\.pt',path.name)
                assert match,(rel,'unexpected event name')
                item=tuple(map(int,match.groups()));assert item not in seen,(case['id'],item);seen.add(item)
                c,l,step,h,tile=item
                assert l in LAYERS and h in HEADS and c in (2,5,8) and step in (0,2)
                t0=(17*l+13*c+7*step+case['seed'])%72
                assert tile in (t0,(t0+36)%72)
                assert (path.parent/f'c{c}_l{l}_s{step}_h{h}_kv.pt').is_file()
            assert len(seen)==4176
            counts.update(rec['coverage'])
        assert len(counts)==348 and set(counts.values())==({96} if split=='train' else {24})
        result[split]=dict(events=len(paths),coverage=dict(counts))
    atomic(OUT/'dataset_audit.json',dict(passed=True,splits=result,test_accessed=False))
    return result

def export(seed):
    root=OUT/'training'/f'C_s{seed}';complete=json.loads((root/'complete.json').read_text())
    assert complete['frozen_exact'] and len(complete['coverage'])==348 and set(complete['coverage'].values())=={192}
    choices=[c for c in complete['checks'] if c['initial_ratio']<=1.02]
    if not choices:raise RuntimeError('No full-coverage C checkpoint passes development initial NMSE gate')
    chosen=min(choices,key=lambda c:c['initial_ratio']);epoch=chosen['epoch']
    ck=torch.load(root/f'epoch{epoch}.pt',map_location='cpu',weights_only=False)
    assert len(ck['coverage'])==348 and set(ck['coverage'].values())=={96*(epoch+1)}
    assert len(ck['gradient_coverage'])==348*3 and min(ck['gradient_coverage'].values())>0
    rows=json.loads((root/f'dev_epoch{epoch}.json').read_text())
    assert len(rows)==8352 and len({r['key'] for r in rows})==348
    state=torch.load(PRE,map_location='cpu',weights_only=False)['model']
    for l in LAYERS:
        for name in ('pool_k','pool_v','pool_feature'):
            state[f'layers.{l}.{name}']=torch.cat([ck['model'][f'layers.l{l}h{h}.{name}'] for h in HEADS],0)
    # Validate every exported position against its trained source, and everything else frozen.
    original=torch.load(PRE,map_location='cpu',weights_only=False)['model']
    for k,v in state.items():
        if k.rsplit('.',1)[-1] not in ('pool_k','pool_v','pool_feature'):assert torch.equal(v,original[k]),k
    output=dict(model=state,coverage=[(l,h) for l in LAYERS for h in HEADS],seed=seed,epoch=epoch,
                algorithm='full-coverage C+Risk; E48 Full_Pset not yet adapted',
                training_events_per_position=96*(epoch+1),source=digest(root/f'epoch{epoch}.pt'),protocol_sha=digest(OUT/'protocol.json'),development=chosen)
    atomic(root/'deployment.pt',output,True)
    atomic(root/'export.json',dict(passed=True,positions=348,epoch=epoch,sha256=digest(root/'deployment.pt'),full_pset_ready=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export-seed',type=int,choices=[17,29]);a=p.parse_args()
    if a.export_seed is None:audit_data()
    else:export(a.export_seed)
